import torch
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from convlstm import (
    load_and_preprocess_data,
    load_scaler_params,
    AttentionConvLSTM,  # Change to new model
    EnhancedTrendAwareLoss,
    SSHDataset
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
        with tqdm(total=2, desc="    Denormalizing data") as pbar:
            y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
            pbar.update(1)
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
            pbar.update(1)

    valid_idx = mask.flatten() == 1
    y_true_valid = y_true.flatten()[valid_idx]
    y_pred_valid = y_pred.flatten()[valid_idx]

    with tqdm(total=4, desc="    Calculating basic metrics") as pbar:
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

    with tqdm(total=1, desc="    Calculating trend metrics") as pbar:
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
                loss = criterion(output, batch_y, batch_mask)
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

    with tqdm(total=2, desc=f"    Processing data for time point {time_point}") as pbar:
        if scaler:
            pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
            target = scaler.inverse_transform(target.reshape(-1, 1)).reshape(target.shape)
        pbar.update(1)

        vmin = min(pred[time_point].min(), target[time_point].min())
        vmax = max(pred[time_point].max(), target[time_point].max())
        pbar.update(1)

    fig = plt.figure(figsize=(20, 4 * num_steps))
    gs = plt.GridSpec(num_steps, 4, width_ratios=[1, 1, 1, 0.05])

    for i in tqdm(range(num_steps), desc=f"    Plotting images for time point {time_point}"):
        pred_step = pred[time_point, i]
        target_step = target[time_point, i]
        mask_step = mask[time_point, i]
        error = (pred_step - target_step) * mask_step

        pred_masked = np.ma.masked_array(pred_step, ~mask_step.astype(bool))
        target_masked = np.ma.masked_array(target_step, ~mask_step.astype(bool))
        error_masked = np.ma.masked_array(error, ~mask_step.astype(bool))

        for j, (data, title) in enumerate([
            (pred_masked, 'Predicted'),
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

    with tqdm(total=1, desc=f"    Saving image for time point {time_point}") as pbar:
        plt.savefig(f'{save_path}/prediction_sequence_{time_point}.png', dpi=300, bbox_inches='tight')
        plt.close()
        pbar.update(1)


def main():
    print("\nStarting prediction program execution...")

    # Parameter settings
    params = {
        'model_path': 'results/best_model.pth',
        'data_path': r"E:\data\sshnow.mat",
        'input_length': 21,
        'prediction_length': 7,
        'batch_size': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Print device information
    print(f"\nUsing device: {params['device']}")
    if params['device'] == 'cuda':
        print(f"GPU model: {torch.cuda.get_device_name(0)}")

    # Load data and model
    print("\n1. Data preparation phase")
    with tqdm(total=5, desc="Loading data and model") as pbar:
        # Load data
        sequences, targets, masks, _ = load_and_preprocess_data(
            params['data_path'],
            input_length=params['input_length'],
            prediction_length=params['prediction_length'],
            normalize=True,
            training=False
        )
        pbar.update(1)

        # Load normalizer
        scaler = load_scaler_params()
        pbar.update(1)

        # Create dataset and loader
        dataset = SSHDataset(sequences, targets, masks)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=False
        )
        pbar.update(1)

        # Initialize model
        print("\n2. Model initialization phase")
        model = AttentionConvLSTM(
            input_dim=1,
            hidden_dim=64,
            kernel_size=3,
            num_layers=1,
            prediction_length=params['prediction_length'],
            batch_first=True
        ).to(params['device'])
        pbar.update(1)

        criterion = EnhancedTrendAwareLoss().to(params['device'])
        pbar.update(1)

    # Load model weights
    print("\n3. Loading model weights")
    with tqdm(total=1, desc="Loading model checkpoint") as pbar:
        checkpoint = torch.load(params['model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        pbar.update(1)

    # Prediction phase
    print("\n4. Executing prediction phase")
    start_time = time.time()
    predictions, targets, masks, test_loss = predict(
        model, data_loader, criterion, params['device']
    )
    prediction_time = time.time() - start_time

    # Calculate evaluation metrics
    print("\n5. Calculating evaluation metrics phase")
    all_metrics = {}
    for step in tqdm(range(params['prediction_length']), desc="Calculating metrics by time step"):
        print(f"\nCalculating metrics for step {step + 1}:")
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

    # Save results
    print("\n6. Saving results phase")
    save_path = 'predictions'
    os.makedirs(save_path, exist_ok=True)

    with tqdm(total=3, desc="Processing and saving results") as pbar:
        # Denormalize results
        if scaler:
            predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
            targets_orig = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
        else:
            predictions_orig, targets_orig = predictions, targets
        pbar.update(1)

        # Prepare save dictionary
        save_dict = {
            'predictions': predictions_orig,
            'targets': targets_orig,
            'masks': masks,
            'metrics': all_metrics,
            'test_loss': test_loss
        }
        pbar.update(1)

        # Save to file
        savemat(f'{save_path}/predictions.mat', save_dict)
        pbar.update(1)

    # Generate visualizations
    print("\n7. Generating visualization phase")
    with tqdm(total=2, desc="Generating visualizations") as pbar:
        # Prediction sequence visualization
        for i in range(min(5, len(predictions))):
            visualize_prediction_sequence(
                predictions, targets, masks,
                time_point=i,
                scaler=scaler,
                save_path=save_path
            )
        pbar.update(1)

        # Metrics over time plot
        plt.figure(figsize=(15, 10))
        for metric_name, values in all_metrics.items():
            plt.plot(range(1, params['prediction_length'] + 1),
                     values, marker='o', label=metric_name)
        plt.xlabel('Prediction Step')
        plt.ylabel('Metric Value')
        plt.title('Metrics Change Over Prediction Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_path}/metrics_over_time.png')
        plt.close()
        pbar.update(1)

    # Print result summary
    print("\n8. Execution completed")
    print(f"\nResults saved to: {save_path}/")
    print(f"Total prediction time: {prediction_time:.2f} seconds")
    print(f"Average prediction time per sample: {prediction_time / len(predictions):.4f} seconds")

    # Print detailed metrics
    print("\nPrediction metrics summary:")
    for metric_name, values in all_metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Average: {np.mean(values):.4f}")
        print(f"  Maximum: {np.max(values):.4f}")
        print(f"  Minimum: {np.min(values):.4f}")


if __name__ == "__main__":
    main()