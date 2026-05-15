import torch

import numpy as np

from tqdm import tqdm

import time

import matplotlib.pyplot as plt

from scipy.io import savemat

import os

from sklearn.metrics import r2_score, mean_squared_error



from convlstm_v2 import (

    load_and_preprocess_data,

    load_scaler_params,

    AttentionConvLSTM,

    EnhancedTrendAwareLoss,

    SSHDataset

)





                                                              

      

                                                              



def calculate_trend_metrics(y_true, y_pred, mask):

                                    

                  

    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()

    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()

    if torch.is_tensor(mask):   mask   = mask.cpu().numpy()



    true_trend  = np.diff(y_true, axis=1)

    pred_trend  = np.diff(y_pred, axis=1)

    trend_mask  = mask[:, 1:] * mask[:, :-1]



    valid_idx       = trend_mask.flatten() == 1

    true_trend_v    = true_trend.flatten()[valid_idx]

    pred_trend_v    = pred_trend.flatten()[valid_idx]



    return {

        'Trend_MSE':          mean_squared_error(true_trend_v, pred_trend_v),

        'Trend_Correlation':  np.corrcoef(true_trend_v, pred_trend_v)[0, 1],

        'Direction_Accuracy': np.mean(np.sign(true_trend_v) == np.sign(pred_trend_v))

    }





def calculate_metrics(y_true, y_pred, mask, scaler=None):

                          

    if scaler is not None:

        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)

        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)



    valid_idx  = mask.flatten() == 1

    y_true_v   = y_true.flatten()[valid_idx]

    y_pred_v   = y_pred.flatten()[valid_idx]



    mse  = mean_squared_error(y_true_v, y_pred_v)

    base = {

        'MSE':         mse,

        'RMSE':        np.sqrt(mse),

        'R2':          r2_score(y_true_v, y_pred_v),

        'Correlation': np.corrcoef(y_true_v, y_pred_v)[0, 1]

    }

    trend = calculate_trend_metrics(y_true, y_pred, mask)

    return {**base, **trend}





                                                              

    

                                                              



def predict(model, data_loader, criterion, device='cuda'):

                                      

    model.eval()

    all_preds, all_targets, all_masks = [], [], []

    total_loss = 0.0



    with torch.no_grad():

        for batch_x, batch_y, batch_mask in tqdm(data_loader, desc='Running inference'):

            batch_x    = batch_x.to(device)

            batch_y    = batch_y.to(device)

            batch_mask = batch_mask.to(device)



            output, _ = model(batch_x)



            loss = criterion(output, batch_y, batch_mask)

            total_loss += loss.item()



            all_preds.append(output.cpu())

            all_targets.append(batch_y.cpu())

            all_masks.append(batch_mask.cpu())



    predictions = torch.cat(all_preds,   dim=0).numpy()

    targets     = torch.cat(all_targets, dim=0).numpy()

    masks       = torch.cat(all_masks,   dim=0).numpy()

    avg_loss    = total_loss / len(data_loader)



    return predictions, targets, masks, avg_loss





                                                              

     

                                                              



def visualize_prediction_sequence(pred, target, mask, time_point,

                                   scaler=None, num_steps=7,

                                   save_path='predictions_gulf'):

                                      

    os.makedirs(save_path, exist_ok=True)



    if scaler is not None:

        pred   = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)

        target = scaler.inverse_transform(target.reshape(-1, 1)).reshape(target.shape)



    vmin = min(pred[time_point].min(), target[time_point].min())

    vmax = max(pred[time_point].max(), target[time_point].max())



    fig = plt.figure(figsize=(20, 4 * num_steps))

    gs  = plt.GridSpec(num_steps, 4, width_ratios=[1, 1, 1, 0.05])



    for i in range(num_steps):

        pred_step   = pred[time_point, i]

        target_step = target[time_point, i]

        mask_step   = mask[time_point, i]

        error       = (pred_step - target_step) * mask_step



        pred_m   = np.ma.masked_array(pred_step,   ~mask_step.astype(bool))

        target_m = np.ma.masked_array(target_step, ~mask_step.astype(bool))

        error_m  = np.ma.masked_array(error,        ~mask_step.astype(bool))



        for j, (data, title) in enumerate([(pred_m,   'Prediction'),

                                            (target_m, 'Target'),

                                            (error_m,  'Error')]):

            ax = plt.subplot(gs[i, j])

            if j == 2:

                max_err = np.abs(error_m).max()

                im = ax.imshow(data, cmap='RdBu_r', vmin=-max_err, vmax=max_err)

            else:

                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)

            ax.set_title(f'Step {i + 1} - {title}')

            ax.axis('off')



        if i == 0:

            plt.colorbar(im, cax=plt.subplot(gs[i, 3]))



    plt.suptitle(f'Prediction sequence analysis - time point {time_point}', y=1.01)

    plt.tight_layout()

    save_file = f'{save_path}/prediction_sequence_{time_point}.png'

    plt.savefig(save_file, dpi=150, bbox_inches='tight')

    plt.close()

    print(f'  Saved: {save_file}')





def plot_metrics_over_time(all_metrics, prediction_length, save_path='predictions_gulf'):

                         

    os.makedirs(save_path, exist_ok=True)

    steps = range(1, prediction_length + 1)



    plt.figure(figsize=(15, 10))

    for metric_name, values in all_metrics.items():

        plt.plot(steps, values, marker='o', label=metric_name)

    plt.xlabel('Prediction step')

    plt.ylabel('Metric value')

    plt.title('Metrics over prediction steps')

    plt.legend()

    plt.grid(True)

    save_file = f'{save_path}/metrics_over_time.png'

    plt.savefig(save_file, dpi=150)

    plt.close()

    print(f'  Saved: {save_file}')





                                                              

     

                                                              



def main():

    print("=" * 50)

    print("   AttentionConvLSTM inference program")

    print("=" * 50)



                                

    params = {

        'model_path':        'results/best_mode_cv.pth',

        'data_path':         r"E:\Ocean modelling\data\sshfuture.mat",

        'scaler_path':       'scaler_params.json',

        'input_length':      21,

        'prediction_length': 7,

        'batch_size':        8,

        'save_path':         'predictions_dam_now',

        'device':            'cuda' if torch.cuda.is_available() else 'cpu',

                               

        'hidden_dim':        64,

        'kernel_size':       3,

        'num_layers':        2,

    }



    device = params['device']

    print(f"\nDevice: {device}")

    if device == 'cuda':

        print(f"GPU model:  {torch.cuda.get_device_name(0)}")



                                   

    print("\n[1/6] Loading and preprocessing data...")

    sequences, targets, masks, _ = load_and_preprocess_data(

        params['data_path'],

        input_length=params['input_length'],

        prediction_length=params['prediction_length'],

        normalize=True,

        training=False

    )

    print(f"  sequences shape : {sequences.shape}")

    print(f"  targets   shape : {targets.shape}")

    print(f"  masks     shape : {masks.shape}")



                                     

    print("\n[2/6] Loading scaler...")

    scaler = load_scaler_params(params['scaler_path'])

    print(f"  Scaler range: {scaler.feature_range}")



                                            

    print("\n[3/6] Building DataLoader...")

    dataset     = SSHDataset(sequences, targets, masks)

    data_loader = torch.utils.data.DataLoader(

        dataset,

        batch_size=params['batch_size'],

        shuffle=False

    )

    print(f"  Total samples: {len(dataset)}")



                                       

    print("\n[4/6] Initializing model and loading weights...")

    model = AttentionConvLSTM(

        input_dim=1,

        hidden_dim=params['hidden_dim'],

        kernel_size=params['kernel_size'],

        num_layers=params['num_layers'],

        prediction_length=params['prediction_length'],

        batch_first=True

    ).to(device)



    criterion = EnhancedTrendAwareLoss().to(device)



    checkpoint = torch.load(params['model_path'], map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])



    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Model parameters: {total_params:,}")

    print(f"  Source epoch: {checkpoint.get('epoch', 'N/A')}")

    print(f"  Validation loss:   {checkpoint.get('val_loss', 'N/A'):.6f}")



                                   

    print("\n[5/6] Running inference...")

    t0 = time.time()

    predictions, targets_np, masks_np, avg_loss = predict(

        model, data_loader, criterion, device

    )

    elapsed = time.time() - t0



    print(f"  Inference completed, {len(predictions)} samples")

    print(f"  Average loss:   {avg_loss:.6f}")

    print(f"  Total time:     {elapsed:.2f}s  ({elapsed / len(predictions) * 1000:.2f} ms/sample)")



                                   

    print("\n[6/6] Calculating metrics...")

    all_metrics = {}

    for step in range(params['prediction_length']):

        step_metrics = calculate_metrics(

            targets_np[:, step, ...],

            predictions[:, step, ...],

            masks_np[:, step, ...],

            scaler

        )

        for k, v in step_metrics.items():

            all_metrics.setdefault(k, []).append(v)

        print(f"  Step {step + 1}: RMSE={step_metrics['RMSE']:.4f}  "

              f"R2={step_metrics['R2']:.4f}  "

              f"Dir_Acc={step_metrics['Direction_Accuracy']:.4f}")



                                

    save_path = params['save_path']

    os.makedirs(save_path, exist_ok=True)



    pred_orig   = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)

    target_orig = scaler.inverse_transform(targets_np.reshape(-1, 1)).reshape(targets_np.shape)



    savemat(f'{save_path}/predictions_future.mat', {

        'predictions': pred_orig,

        'targets':     target_orig,

        'masks':       masks_np,

        'metrics':     str(all_metrics),

        'avg_loss':    avg_loss

    })

    print(f"\nResults saved to: {save_path}/predictions.mat")



                     

    print("\nGenerating spatial visualizations for the first 5 time points...")

    for i in tqdm(range(min(5, len(predictions))), desc='Plotting'):

        visualize_prediction_sequence(

            predictions, targets_np, masks_np,

            time_point=i,

            scaler=scaler,

            num_steps=params['prediction_length'],

            save_path=save_path

        )



                 

    plot_metrics_over_time(all_metrics, params['prediction_length'], save_path)



                                

    print("\n" + "=" * 50)

    print("Prediction metric summary (mean over steps)")

    print("=" * 50)

    for metric_name, values in all_metrics.items():

        best = np.min(values) if 'MSE' in metric_name else np.max(values)

        print(f"  {metric_name:<22} mean={np.mean(values):.4f}  best={best:.4f}")

    print("=" * 50)





if __name__ == "__main__":

    main()
