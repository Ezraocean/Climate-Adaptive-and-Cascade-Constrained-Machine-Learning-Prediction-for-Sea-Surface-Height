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
import random

              
from CCM_train_clean import (
    load_and_preprocess_data,
    create_sliding_windows,
    load_scaler_params,
    SSHPhysicsGuidedConvLSTM,
    SSHDataset,
    load_lonlat_data,
    set_random_seed,
    set_worker_seed
)


def calculate_trend_metrics(y_true, y_pred, mask):
    """Calculate trend-related evaluation metrics."""
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
        'Trend_Correlation': np.corrcoef(true_trend_valid, pred_trend_valid)[0, 1] if len(
            true_trend_valid) > 1 else 0.0,
        'Direction_Accuracy': np.mean(np.sign(true_trend_valid) == np.sign(pred_trend_valid)) if len(
            true_trend_valid) > 0 else 0.0
    }

    return trend_metrics


def calculate_metrics(y_true, y_pred, mask, scaler=None):
    """Calculate evaluation metrics."""
    if scaler:
        with tqdm(total=2, desc="    Denormalizing data") as pbar:
                      
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

    if len(y_true_valid) == 0:
        return {
            'MSE': float('nan'),
            'RMSE': float('nan'),
            'R2': float('nan'),
            'Correlation': float('nan'),
            'Trend_MSE': float('nan'),
            'Trend_Correlation': float('nan'),
            'Direction_Accuracy': float('nan')
        }

    with tqdm(total=4, desc="    Computing base metrics") as pbar:
        mse = mean_squared_error(y_true_valid, y_pred_valid)
        pbar.update(1)
        rmse = np.sqrt(mse)
        pbar.update(1)
        r2 = r2_score(y_true_valid, y_pred_valid)
        pbar.update(1)
        correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1] if len(y_true_valid) > 1 else 0.0
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
    """Prediction function."""
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    total_loss = 0.0 if criterion else None

    with torch.no_grad():
        for batch_x, batch_y, batch_mask in tqdm(data_loader, desc='Running prediction'):
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
    """Enhanced prediction-sequence visualization."""
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

    for i in tqdm(range(num_steps), desc=f"    Plotting time point {time_point} image"):
        pred_step = pred[time_point, i]
        target_step = target[time_point, i]
        mask_step = mask[time_point, i]
        error = (pred_step - target_step) * mask_step

        pred_masked = np.ma.masked_array(pred_step, ~mask_step.astype(bool))
        target_masked = np.ma.masked_array(target_step, ~mask_step.astype(bool))
        error_masked = np.ma.masked_array(error, ~mask_step.astype(bool))

        for j, (data, title) in enumerate([
            (pred_masked, 'Prediction'),
            (target_masked, 'Target'),
            (error_masked, 'Error')
        ]):
            ax = plt.subplot(gs[i, j])
            if j == 2:
                max_error = np.abs(error_masked).max()
                if max_error > 0:
                    im = ax.imshow(data, cmap='RdBu_r', vmin=-max_error, vmax=max_error)
                else:
                    im = ax.imshow(data, cmap='RdBu_r')
            else:
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Step {i + 1} - {title}')

        if i == 0:
            plt.colorbar(im, cax=plt.subplot(gs[i, 3]))

    plt.suptitle(f'Prediction sequence analysis - time point {time_point}', y=1.02)
    plt.tight_layout()

    with tqdm(total=1, desc=f"    Saving time point {time_point} image") as pbar:
        plt.savefig(f'{save_path}/prediction_sequence_{time_point}.png', dpi=300, bbox_inches='tight')
        plt.close()
        pbar.update(1)


def visualize_energy_cascade_features(model, predictions, targets, masks, save_path, num_samples=3):
    """Visualize energy cascade features."""
    print("\nComputing and visualizing energy cascade features...")

    if not hasattr(model, 'energy_cascade') or model.energy_cascade is None:
        print("The model has no energy cascade module; skipping this visualization")
        return

    with torch.no_grad():
                  
        sample_idx = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

        fig, axes = plt.subplots(len(sample_idx), 3, figsize=(15, 5 * len(sample_idx)))
        if len(sample_idx) == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(sample_idx):
            try:
                                 
                pred_ssh = torch.tensor(predictions[idx:idx + 1, -1:]).cuda()                
                target_ssh = torch.tensor(targets[idx:idx + 1, -1:]).cuda()                

                          
                if pred_ssh.dim() == 3:
                    pred_ssh = pred_ssh.unsqueeze(1)          
                if target_ssh.dim() == 3:
                    target_ssh = target_ssh.unsqueeze(1)

                          
                pred_cascade, pred_u, pred_v = model.energy_cascade(pred_ssh)
                target_cascade, target_u, target_v = model.energy_cascade(target_ssh)

                        
                pred_speed = torch.sqrt(pred_u ** 2 + pred_v ** 2).cpu().numpy()[0, 0]
                target_speed = torch.sqrt(target_u ** 2 + target_v ** 2).cpu().numpy()[0, 0]

                        
                ax1, ax2, ax3 = axes[i]

                im1 = ax1.imshow(pred_speed, cmap='viridis')
                ax1.set_title(f'samples {idx} - predicted speed magnitude')
                plt.colorbar(im1, ax=ax1)

                im2 = ax2.imshow(target_speed, cmap='viridis')
                ax2.set_title(f'samples {idx} - target speed magnitude')
                plt.colorbar(im2, ax=ax2)

                      
                error = pred_speed - target_speed
                vmax = np.max(np.abs(error)) if np.max(np.abs(error)) > 0 else 1
                im3 = ax3.imshow(error, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax3.set_title('Speed difference')
                plt.colorbar(im3, ax=ax3)

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                            
                for ax in axes[i]:
                    ax.text(0.5, 0.5, f'Error processing\nsample {idx}',
                            ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(f'{save_path}/geostrophic_velocity.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    print("\n🌊 Starting SSH physics-guided model prediction...")

                    
    random_seed = 42
    set_random_seed(random_seed)
    print(f"Random seed set to: {random_seed}")

                     
    params = {
        'model_path': 'results/best_model.pth',             
        'data_path': r"E:\Ocean modelling\data\sshfuture.mat",
        'input_length': 21,
        'prediction_length': 7,
        'batch_size': 8,            
        'save_path': 'predictions_ccm',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

            
    if not os.path.exists(params['model_path']):
        print(f"❌ Model file does not exist: {params['model_path']}")
        print("Please check whether the model path is correct")
        return

            
    print(f"\n🖥️  Device: {params['device']}")
    if params['device'] == 'cuda':
        print(f"GPU model: {torch.cuda.get_device_name(0)}")

                    
    f0 = 8.4e-5
    g = 9.8
    rho0 = 1027.4

             
    print("\n📊 1. Data preparation stage")
    print("🌍 Loading lon/lat data...")
    lon_grid, lat_grid = load_lonlat_data()
    if lon_grid is None or lat_grid is None:
        print("Unable to load lon/lat data; exiting prediction")
        return

            
    scaler_path = os.path.dirname(params['model_path']) + "/scaler_params.json"
    if os.path.exists(scaler_path):
        scaler = load_scaler_params(scaler_path)
        print("✅ Scaler loaded successfully")
    else:
        print(f"⚠️  Scaler file does not exist: {scaler_path}，trying the default path")
        scaler = load_scaler_params()

                           
    ssh_data, mask_data, _ = load_and_preprocess_data(
        params['data_path'],
        scaler=scaler,
        input_length=params['input_length'],
        prediction_length=params['prediction_length'],
        normalize=True,
        training=False
    )

              
    sequences, targets, masks = create_sliding_windows(
        ssh_data, mask_data, params['input_length'], params['prediction_length']
    )
    print(f"Number of generated samples: {len(sequences)}")

               
    dataset = SSHDataset(sequences, targets, masks)
    rng = torch.Generator()
    rng.manual_seed(random_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=rng
    )

                        
    print("\n🧠 2. Model initialization stage")
    model = SSHPhysicsGuidedConvLSTM(
        input_dim=1,
        hidden_dim=64,
        kernel_size=3,
        num_layers=2,
        prediction_length=params['prediction_length'],
        batch_first=True,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        dx=1000, dy=1000,
        f0=f0, g=g, rho0=rho0
    ).to(params['device'])

            
    print("\n💾 3. Loading model weights")
    checkpoint = torch.load(params['model_path'], map_location=params['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model weights loaded successfully (Fold {checkpoint.get('fold', '?')}, Epoch {checkpoint['epoch']})")
    print(f"📈 Training loss: {checkpoint['train_loss']:.4f}")
    print(f"📉 Validation loss: {checkpoint['val_loss']:.4f}")
    criterion = None              

          
    print("\n4. Prediction stage")
    start_time = time.time()
    predictions, targets, masks, test_loss = predict(
        model, data_loader, None, params['device']
    )
    prediction_time = time.time() - start_time
    print(f"⏱️  Prediction completed, elapsed time: {prediction_time:.2f} seconds")
    if test_loss is not None:
        print(f"📊 Test loss: {test_loss:.4f}")

            
    print("\n5. Evaluation metric computation stage")
    all_metrics = {}
    for step in tqdm(range(params['prediction_length']), desc="Computing metrics by forecast step"):
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

            
    print("\n📊 Prediction metric summary:")
    for metric_name, values in all_metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            print(f"\n{metric_name}:")
            for step_i, v in enumerate(values):
                print(f"  Step {step_i+1}: {v:.6f}")
            print(f"  Mean: {np.mean(valid_values):.6f}")

          
    print("\n💾 6. Saving results stage")
    save_path = params.get('save_path', 'predictions_ccm')
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
            'metrics': str(all_metrics),
            'test_loss': test_loss if test_loss is not None else 0.0,
            'prediction_time': prediction_time
        }
        pbar.update(1)

               
        savemat(f'{save_path}/predictions_future.mat', save_dict)

                     
        metrics_dict = {k: [float(v) if not np.isnan(v) else None for v in vals]
                        for k, vals in all_metrics.items()}
        with open(f'{save_path}/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        pbar.update(1)

           
    print("\n🎨 7. Visualization stage")
    with tqdm(total=3, desc="Generating visualizations") as pbar:
                 
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
            'Accuracy metrics': ['MSE', 'RMSE', 'R2', 'Correlation'],
            'Trend metrics': ['Trend_MSE', 'Trend_Correlation', 'Direction_Accuracy']
        }

        fig, axes = plt.subplots(len(metrics_groups), 1, figsize=(12, 8 * len(metrics_groups)))
        if len(metrics_groups) == 1:
            axes = [axes]

        for i, (group_name, metrics_list) in enumerate(metrics_groups.items()):
            ax = axes[i]
            for metric_name in metrics_list:
                if metric_name in all_metrics:
                    values = [v for v in all_metrics[metric_name] if not np.isnan(v)]
                    if values:
                        ax.plot(range(1, len(values) + 1), values,
                                marker='o', label=metric_name)
            ax.set_xlabel('Forecast step')
            ax.set_ylabel('Metric value')
            ax.set_title(f'{group_name} over forecast steps')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        pbar.update(1)

                   
        visualize_energy_cascade_features(model, predictions, targets, masks, save_path)
        pbar.update(1)

            
    print("\n🎉 8. Execution completed")
    print(f"📁 Results saved to: {save_path}/")
    print(f"⏱️  Total prediction time: {prediction_time:.2f} seconds")
    print(f"Average prediction time per sample: {prediction_time / len(predictions):.4f} seconds")

            
    print("\n📊 Prediction metric summary:")
    for metric_name, values in all_metrics.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            print(f"\n{metric_name}:")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Maximum: {np.max(valid_values):.4f}")
            print(f"  Minimum: {np.min(valid_values):.4f}")
            print(f"  Standard deviation: {np.std(valid_values):.4f}")
        else:
            print(f"\n{metric_name}: No valid data")

            
    summary = {
        'total_prediction_time': prediction_time,
        'avg_time_per_sample': prediction_time / len(predictions),
        'num_samples': len(predictions),
        'test_loss': test_loss,
        'model_path': params['model_path'],
        'random_seed': random_seed
    }

    with open(f'{save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ All results have been saved！")


if __name__ == "__main__":
    main()