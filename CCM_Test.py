import torch
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
import json
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

from KEcascadeconvLSTM_v5_modified import (
    load_and_preprocess_data,
    load_scaler_params,
    SSHPhysicsGuidedConvLSTM,
    BatchOptimizedSSHPhysicsConstrainedLoss,
    SSHDataset,
    visualize_predictions_english
)


def calculate_trend_metrics(y_true, y_pred, mask):
    """Calculate trend-related evaluation metrics"""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    true_trend = np.diff(y_true, axis=1)
    pred_trend = np.diff(y_pred, axis=1)
    trend_mask = mask[:, 1:] * mask[:, :-1]

    valid_idx = trend_mask.flatten() == 1
    true_trend_valid = true_trend.flatten()[valid_idx]
    pred_trend_valid = pred_trend.flatten()[valid_idx]

    trend_metrics = {
        'Trend_MSE': mean_squared_error(true_trend_valid, pred_trend_valid),
        'Trend_Correlation': np.corrcoef(true_trend_valid, pred_trend_valid)[0, 1],
        'Direction_Accuracy': np.mean(np.sign(true_trend_valid) == np.sign(pred_trend_valid))
    }

    return trend_metrics


def calculate_metrics(y_true, y_pred, mask, scaler=None):
    """Calculate evaluation metrics"""
    if scaler:
        with tqdm(total=2, desc="    Inverse normalizing data") as pbar:
            def inverse_normalize(data):
                shape = data.shape
                data_flat = data.reshape(-1, 1)
                data_inverse = scaler.inverse_transform(data_flat)
                return data_inverse.reshape(shape)

            y_true = inverse_normalize(y_true)
            pbar.update(1)
            y_pred = inverse_normalize(y_pred)
            pbar.update(1)

    valid_idx = mask.flatten() == 1
    y_true_valid = y_true.flatten()[valid_idx]
    y_pred_valid = y_pred.flatten()[valid_idx]

    with tqdm(total=4, desc="    Computing basic metrics") as pbar:
        mse = mean_squared_error(y_true_valid, y_pred_valid)
        pbar.update(1)
        rmse = np.sqrt(mse)
        pbar.update(1)
        r2 = r2_score(y_true_valid, y_pred_valid)
        pbar.update(1)
        correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
        pbar.update(1)

    base_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Correlation': correlation
    }

    with tqdm(total=1, desc="    Computing trend metrics") as pbar:
        trend_metrics = calculate_trend_metrics(y_true, y_pred, mask)
        pbar.update(1)

    return {**base_metrics, **trend_metrics}


def predict(model, data_loader, criterion=None, device='cuda'):
    """Prediction function"""
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    total_loss = 0.0 if criterion else None

    with torch.no_grad():
        for batch_x, batch_y, batch_mask in tqdm(data_loader, desc='Executing prediction'):
            batch_x, batch_y, batch_mask = (
                batch_x.to(device),
                batch_y.to(device),
                batch_mask.to(device)
            )

            output, _ = model(batch_x)

            if criterion:
                loss = criterion(output, batch_y, batch_mask, model)
                total_loss += loss.item()

            all_preds.append(output.cpu())
            all_targets.append(batch_y.cpu())
            all_masks.append(batch_mask.cpu())

    with tqdm(total=3, desc='Processing prediction results') as pbar:
        predictions = torch.cat(all_preds, dim=0).numpy()
        pbar.update(1)
        targets = torch.cat(all_targets, dim=0).numpy()
        pbar.update(1)
        masks = torch.cat(all_masks, dim=0).numpy()
        pbar.update(1)

    avg_loss = total_loss / len(data_loader) if criterion else None
    return predictions, targets, masks, avg_loss


def visualize_prediction_sequence(pred, target, mask, time_point, scaler=None, num_steps=7, save_path='predictions'):
    """Enhanced prediction sequence visualization"""
    os.makedirs(save_path, exist_ok=True)

    with tqdm(total=2, desc=f"    Processing time point {time_point} data") as pbar:
        if scaler:
            def inverse_normalize(data):
                shape = data.shape
                data_flat = data.reshape(-1, 1)
                data_inverse = scaler.inverse_transform(data_flat)
                return data_inverse.reshape(shape)

            pred = inverse_normalize(pred)
            target = inverse_normalize(target)
        pbar.update(1)

        vmin = min(pred[time_point].min(), target[time_point].min())
        vmax = max(pred[time_point].max(), target[time_point].max())
        pbar.update(1)

    fig = plt.figure(figsize=(20, 4 * num_steps))
    gs = plt.GridSpec(num_steps, 4, width_ratios=[1, 1, 1, 0.05])

    for i in tqdm(range(num_steps), desc=f"    Drawing time point {time_point} images"):
        pred_step = pred[time_point, i]
        target_step = target[time_point, i]
        mask_step = mask[time_point, i]
        error = (pred_step - target_step) * mask_step

        pred_masked = np.ma.masked_array(pred_step, ~mask_step.astype(bool))
        target_masked = np.ma.masked_array(target_step, ~mask_step.astype(bool))
        error_masked = np.ma.masked_array(error, ~mask_step.astype(bool))

        for j, (data, title) in enumerate([
            (pred_masked, 'Prediction'),
            (target_masked, 'Ground Truth'),
            (error_masked, 'Error')
        ]):
            ax = plt.subplot(gs[i, j])
            if j == 2:
                max_error = np.abs(error_masked).max()
                im = ax.imshow(data, cmap='RdBu_r', vmin=-max_error, vmax=max_error)
            else:
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Step {i + 1} - {title}')

        if i == 0:
            plt.colorbar(im, cax=plt.subplot(gs[i, 3]))

    plt.suptitle(f'Prediction Sequence Analysis - Time Point {time_point}', y=1.02)
    plt.tight_layout()

    with tqdm(total=1, desc=f"    Saving time point {time_point} image") as pbar:
        plt.savefig(f'{save_path}/prediction_sequence_{time_point}.png', dpi=300, bbox_inches='tight')
        plt.close()
        pbar.update(1)


def load_lonlat_data():
    """Load longitude and latitude data"""
    try:
        import h5py
        with h5py.File('E:/data/lonlat.mat', 'r') as f:
            print("Variables in MATLAB file:", list(f.keys()))

            lon3_raw = f['lon3'][:]
            lat3_raw = f['lat3'][:]

            print(f"Original data shape: lon3={lon3_raw.shape}, lat3={lat3_raw.shape}")

            if lon3_raw.shape[0] == 1:
                lon_1d = lon3_raw.flatten()
            else:
                lon_1d = lon3_raw[:, 0] if lon3_raw.shape[1] == 1 else lon3_raw[0, :]

            if lat3_raw.shape[0] == 1:
                lat_1d = lat3_raw.flatten()
            else:
                lat_1d = lat3_raw[:, 0] if lat3_raw.shape[1] == 1 else lat3_raw[0, :]

            print(f"Flattened shape: lon_1d={lon_1d.shape}, lat_1d={lat_1d.shape}")

            lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing='ij')

            print(f"Successfully loaded lon/lat data")
            print(f"1D arrays: lon length={len(lon_1d)}, lat length={len(lat_1d)}")
            print(f"2D grids: lon_grid={lon_grid.shape}, lat_grid={lat_grid.shape}")
            print(f"Longitude range: {lon_1d.min():.2f}° to {lon_1d.max():.2f}°")
            print(f"Latitude range: {lat_1d.min():.2f}° to {lat_1d.max():.2f}°")

            return lon_grid, lat_grid

    except Exception as e:
        print(f"Failed to load lon/lat data: {e}")
        print("Please ensure file path is correct: E:/data/lonlat.mat")
        print("If still having issues, check MATLAB file format and variable names")
        return None, None


def main():
    print("\nStarting prediction program...")

    params = {
        'model_path': 'results/ssh_enhanced_visualization_20250612_163830/best_model.pth',
        'data_path': r"E:\data\sshfuture.mat",
        'input_length': 21,
        'prediction_length': 7,
        'batch_size': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\nUsing device: {params['device']}")
    if params['device'] == 'cuda':
        print(f"GPU model: {torch.cuda.get_device_name(0)}")

    print("\n1. Data preparation phase")
    with tqdm(total=5, desc="Data and model loading") as pbar:
        sequences, targets, masks, _ = load_and_preprocess_data(
            params['data_path'],
            input_length=params['input_length'],
            prediction_length=params['prediction_length'],
            normalize=True,
            training=False
        )
        pbar.update(1)

        checkpoint = torch.load(params['model_path'])
        model_config = checkpoint['config']
        print(f"\nLoaded model config: {model_config['experiment_name']}")
        pbar.update(1)

        physics_params = model_config['physics_params']
        loss_weights = model_config['loss_weights']

        scaler_path = os.path.dirname(params['model_path']) + "/scaler_params.json"
        scaler = load_scaler_params(scaler_path)
        pbar.update(1)

        dataset = SSHDataset(sequences, targets, masks)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=False
        )
        pbar.update(1)

        print("\n2. Model initialization phase")
        print("Loading lon/lat data...")
        lon_grid, lat_grid = load_lonlat_data()

        if lon_grid is None or lat_grid is None:
            print("Cannot load lon/lat data, exiting prediction")
            return

        model = SSHPhysicsGuidedConvLSTM(
            input_dim=1,
            hidden_dim=model_config["hidden_dim"],
            kernel_size=model_config["kernel_size"],
            num_layers=model_config["num_layers"],
            prediction_length=params['prediction_length'],
            batch_first=True,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            dx=1000,
            dy=1000,
            f0=physics_params["f0"],
            g=physics_params["g"],
            rho0=physics_params["rho0"]
        ).to(params['device'])
        pbar.update(1)

        criterion = BatchOptimizedSSHPhysicsConstrainedLoss(
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            alpha=loss_weights["alpha"],
            beta=loss_weights["beta"],
            gamma=loss_weights["gamma"],
            delta=loss_weights["delta"],
            dt=physics_params["dt"],
            f0=physics_params["f0"],
            g=physics_params["g"],
            rho0=physics_params["rho0"]
        ).to(params['device'])

    print("\n3. Loading model weights")
    with tqdm(total=1, desc="Loading model checkpoint") as pbar:
        model.load_state_dict(checkpoint['model_state_dict'])
        pbar.update(1)

    print("\n4. Executing prediction phase")
    start_time = time.time()
    predictions, targets, masks, test_loss = predict(
        model, data_loader, criterion, params['device']
    )
    prediction_time = time.time() - start_time

    print("\n5. Computing evaluation metrics phase")
    all_metrics = {}
    for step in tqdm(range(params['prediction_length']), desc="Computing metrics by time step"):
        print(f"\nComputing metrics for step {step + 1}:")
        step_metrics = calculate_metrics(
            targets[:, step, ...],
            predictions[:, step, ...],
            masks[:, step, ...],
            scaler
        )
        for metric_name, value in step_metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    print("\n6. Saving results phase")
    save_path = 'predictions'
    os.makedirs(save_path, exist_ok=True)

    with tqdm(total=3, desc="Processing and saving results") as pbar:
        if scaler:
            def inverse_normalize(data):
                shape = data.shape
                data_flat = data.reshape(-1, 1)
                data_inverse = scaler.inverse_transform(data_flat)
                return data_inverse.reshape(shape)

            predictions_orig = inverse_normalize(predictions)
            targets_orig = inverse_normalize(targets)
        else:
            predictions_orig, targets_orig = predictions, targets
        pbar.update(1)

        save_dict = {
            'predictions': predictions_orig,
            'targets': targets_orig,
            'masks': masks,
            'metrics': all_metrics,
            'test_loss': test_loss
        }
        pbar.update(1)

        savemat(f'{save_path}/predictions.mat', save_dict)

        with open(f'{save_path}/metrics.json', 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in all_metrics.items()}, f, indent=4)
        pbar.update(1)

    print("\n7. Generating visualization phase")
    with tqdm(total=2, desc="Generating visualization") as pbar:
        for i in range(min(5, len(predictions))):
            visualize_prediction_sequence(
                predictions, targets, masks,
                time_point=i,
                scaler=scaler,
                save_path=save_path
            )
        pbar.update(1)

        plt.figure(figsize=(15, 10))

        metrics_groups = {
            'Accuracy Metrics': ['MSE', 'RMSE', 'R2', 'Correlation'],
            'Trend Metrics': ['Trend_MSE', 'Trend_Correlation', 'Direction_Accuracy']
        }

        fig, axes = plt.subplots(len(metrics_groups), 1, figsize=(12, 8 * len(metrics_groups)))

        for i, (group_name, metrics_list) in enumerate(metrics_groups.items()):
            ax = axes[i] if len(metrics_groups) > 1 else axes
            for metric_name in metrics_list:
                if metric_name in all_metrics:
                    ax.plot(range(1, params['prediction_length'] + 1),
                            all_metrics[metric_name], marker='o', label=metric_name)
            ax.set_xlabel('Prediction Step')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'{group_name} vs Prediction Step')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_over_time.png', dpi=300)
        plt.close()
        pbar.update(1)

        print("\nComputing energy cascade features...")
        with torch.no_grad():
            sample_idx = np.random.choice(len(predictions), min(3, len(predictions)), replace=False)

            fig, axes = plt.subplots(len(sample_idx), 3, figsize=(15, 5 * len(sample_idx)))

            for i, idx in enumerate(sample_idx):
                pred_ssh = torch.tensor(predictions[idx, -1:]).cuda()
                target_ssh = torch.tensor(targets[idx, -1:]).cuda()

                pred_cascade, pred_u, pred_v = model.energy_cascade(pred_ssh)
                target_cascade, target_u, target_v = model.energy_cascade(target_ssh)

                pred_speed = torch.sqrt(pred_u ** 2 + pred_v ** 2).cpu().numpy()[0, 0]
                target_speed = torch.sqrt(target_u ** 2 + target_v ** 2).cpu().numpy()[0, 0]

                if len(sample_idx) == 1:
                    ax1 = axes[0]
                    ax2 = axes[1]
                    ax3 = axes[2]
                else:
                    ax1 = axes[i, 0]
                    ax2 = axes[i, 1]
                    ax3 = axes[i, 2]

                im1 = ax1.imshow(pred_speed, cmap='viridis')
                ax1.set_title(f'Sample {idx} - Predicted Velocity Magnitude')
                plt.colorbar(im1, ax=ax1)

                im2 = ax2.imshow(target_speed, cmap='viridis')
                ax2.set_title(f'Sample {idx} - True Velocity Magnitude')
                plt.colorbar(im2, ax=ax2)

                error = pred_speed - target_speed
                vmax = np.max(np.abs(error))
                im3 = ax3.imshow(error, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax3.set_title('Velocity Difference')
                plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            plt.savefig(f'{save_path}/geostrophic_velocity.png', dpi=300)
            plt.close()

    print("\n8. Execution completed")
    print(f"\nResults saved to: {save_path}/")
    print(f"Total prediction time: {prediction_time:.2f} seconds")
    print(f"Average prediction time per sample: {prediction_time / len(predictions):.4f} seconds")

    print("\nPrediction metrics summary:")
    for metric_name, values in all_metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Average: {np.mean(values):.4f}")
        print(f"  Maximum: {np.max(values):.4f}")
        print(f"  Minimum: {np.min(values):.4f}")


if __name__ == "__main__":
    main()