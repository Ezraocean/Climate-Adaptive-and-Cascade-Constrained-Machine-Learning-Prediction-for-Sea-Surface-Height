import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np

import h5py

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score, mean_squared_error

from tqdm import tqdm

import time

import warnings

import os

import matplotlib.pyplot as plt

import json

from torch.optim.lr_scheduler import OneCycleLR

import torch.nn.functional as F

import random



os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

warnings.filterwarnings('ignore')





def set_random_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        if hasattr(torch, 'use_deterministic_algorithms'):

            torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to: {seed}")





def set_worker_seed(worker_id):

    worker_seed = torch.initial_seed() % 2 ** 32

    np.random.seed(worker_seed)

    random.seed(worker_seed)





def select_gpu(gpu_id=1):

    if not torch.cuda.is_available():

        print("CUDA is unavailable; using CPU for training")

        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()

    print(f"Detected {num_gpus} GPUs:")

    for i in range(num_gpus):

        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_id >= num_gpus:

        print(f"Warning: specified GPU {gpu_id} does not exist; using GPU 0")

        gpu_id = 0

    device = torch.device(f'cuda:{gpu_id}')

    torch.cuda.set_device(device)

    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    return device





class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv(x)

        return self.sigmoid(x)





class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super(TemporalAttention, self).__init__()

        self.hidden_dim = hidden_dim

        self.attention_layer = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim // 2),

            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1)

        )



    def forward(self, x):

        b, t, c, h, w = x.size()

        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous()

        x_reshaped = x_reshaped.view(-1, t, c)

        attn_weights = self.attention_layer(x_reshaped)

        attn_weights = F.softmax(attn_weights, dim=1)

        attended = torch.bmm(attn_weights.transpose(1, 2), x_reshaped)

        attended = attended.view(b, h, w, c).permute(0, 3, 1, 2)

        return attended





class RefinedTrendAwareLoss(nn.Module):

    def __init__(self, alpha=0.4, beta=0.6, threshold=0.04, weight_multiplier=2.75):

        super(RefinedTrendAwareLoss, self).__init__()

        self.alpha = alpha

        self.beta = beta

        self.threshold = threshold

        self.weight_multiplier = weight_multiplier

        self.mse = nn.MSELoss(reduction='none')



    def forward(self, pred, target, mask):

        mse_loss = self.mse(pred, target)

        pred_diff = pred[:, 1:] - pred[:, :-1]

        target_diff = target[:, 1:] - target[:, :-1]

        trend_mse = self.mse(pred_diff, target_diff)

        change_magnitude = torch.abs(target_diff)

        weight = 1.0 + self.weight_multiplier * (change_magnitude > self.threshold).float()

        combined_trend_loss = trend_mse * (1 + weight)

        trend_mask = mask[:, 1:] * mask[:, :-1]

        loss = (

                self.alpha * (mse_loss * mask).sum() / mask.sum()

                + self.beta * (combined_trend_loss * trend_mask).sum() / trend_mask.sum()

        )

        return loss





class MetricsTracker:

    def __init__(self):

        self.train_losses = []

        self.val_losses = []

        self.learning_rates = []



    def update(self, train_loss, val_loss, lr):

        self.train_losses.append(train_loss)

        self.val_losses.append(val_loss)

        self.learning_rates.append(lr)



    def plot_metrics(self, save_path='results_gulf'):

        os.makedirs(save_path, exist_ok=True)

        epochs = range(1, len(self.train_losses) + 1)



        fig = plt.figure(figsize=(15, 12))

        gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])



        ax1 = plt.subplot(gs[0])

        ax1.semilogy(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)

        ax1.semilogy(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)

        ax1.set_title('Training and Validation Loss (Log Scale)', fontsize=12)

        ax1.set_xlabel('Epochs')

        ax1.set_ylabel('Loss (log scale)')

        ax1.legend()

        ax1.grid(True)



        ax2 = plt.subplot(gs[1])

        ax2.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)

        ax2.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)

        ax2.set_title('Training and Validation Loss (Linear Scale)', fontsize=12)

        ax2.set_xlabel('Epochs')

        ax2.set_ylabel('Loss')

        ax2.legend()

        ax2.grid(True)



        ax3 = plt.subplot(gs[2])

        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)

        ax3.set_title('Learning Rate Schedule', fontsize=12)

        ax3.set_xlabel('Epochs')

        ax3.set_ylabel('Learning Rate')

        ax3.grid(True)



        plt.tight_layout()

        plt.savefig(f'{save_path}/training_metrics.png', dpi=300, bbox_inches='tight')

        plt.close()



        plt.figure(figsize=(10, 6))

        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)

        plt.plot(epochs, loss_diff, 'purple', linewidth=2, label='Val Loss - Train Loss')

        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)

        plt.fill_between(epochs, loss_diff, 0, where=(loss_diff >= 0), color='red', alpha=0.1)

        plt.fill_between(epochs, loss_diff, 0, where=(loss_diff <= 0), color='green', alpha=0.1)

        plt.title('Loss Difference (Validation - Training)', fontsize=12)

        plt.xlabel('Epochs')

        plt.ylabel('Loss Difference')

        plt.legend()

        plt.grid(True)

        plt.savefig(f'{save_path}/loss_difference.png', dpi=300, bbox_inches='tight')

        plt.close()



        plt.figure(figsize=(10, 6))

        train_diff = np.diff(self.train_losses)

        val_diff = np.diff(self.val_losses)

        plt.plot(epochs[1:], train_diff, 'b-', label='Training Loss Change', alpha=0.7)

        plt.plot(epochs[1:], val_diff, 'r-', label='Validation Loss Change', alpha=0.7)

        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        plt.title('Loss Change per Epoch', fontsize=12)

        plt.xlabel('Epochs')

        plt.ylabel('Loss Change')

        plt.legend()

        plt.grid(True)

        plt.savefig(f'{save_path}/loss_dynamics.png', dpi=300, bbox_inches='tight')

        plt.close()





def save_scaler_params(scaler, save_path='scaler_params.json'):

    params = {

        'data_min': scaler.data_min_.tolist(),

        'data_max': scaler.data_max_.tolist(),

        'feature_range': scaler.feature_range

    }

    with open(save_path, 'w') as f:

        json.dump(params, f)





def load_scaler_params(load_path='scaler_params.json'):

    with open(load_path, 'r') as f:

        params = json.load(f)

    scaler = MinMaxScaler(feature_range=tuple(params['feature_range']))

    scaler.data_min_ = np.array(params['data_min'])

    scaler.data_max_ = np.array(params['data_max'])

    scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / (scaler.data_max_ - scaler.data_min_)

    scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_

    return scaler





class AttentionConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, padding):

        super(AttentionConvLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.padding = nn.ReplicationPad2d(padding)

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,

                              out_channels=4 * hidden_dim,

                              kernel_size=kernel_size,

                              padding=0)

        self.spatial_attention = SpatialAttention(kernel_size=7)



    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        spatial_weights = self.spatial_attention(input_tensor)

        input_tensor = input_tensor * spatial_weights

        combined = torch.cat([input_tensor, h_cur], dim=1)

        padded_combined = self.padding(combined)

        combined_conv = self.conv(padded_combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)

        f = torch.sigmoid(cc_f)

        o = torch.sigmoid(cc_o)

        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g

        h_next = o * torch.tanh(c_next)

        return h_next, c_next





class AttentionConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, prediction_length=3, batch_first=True):

        super(AttentionConvLSTM, self).__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size

        self.num_layers = num_layers

        self.batch_first = batch_first

        self.padding = kernel_size // 2

        self.prediction_length = prediction_length

        self.temporal_attention = TemporalAttention(hidden_dim)

        cell_list = []

        for i in range(self.num_layers):

            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim

            cell_list.append(AttentionConvLSTMCell(

                input_dim=cur_input_dim,

                hidden_dim=self.hidden_dim,

                kernel_size=self.kernel_size,

                padding=self.padding

            ))

        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(hidden_dim, prediction_length, kernel_size=3, padding=1)



    def forward(self, x, hidden_state=None):

        b, t, _, h, w = x.size()

        device = x.device

        if hidden_state is None:

            hidden_state = self._init_hidden(b, h, w, device)

        layer_output_list = []

        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            h_state, c = hidden_state[layer_idx]

            output_inner = []

            for t_step in range(cur_layer_input.size(1)):

                h_state, c = self.cell_list[layer_idx](

                    cur_layer_input[:, t_step, :, :, :],

                    (h_state, c)

                )

                output_inner.append(h_state)

            layer_output = torch.stack(output_inner, dim=1)

            attended_output = self.temporal_attention(layer_output)

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)

            last_state_list.append([h_state, c])

        out = self.conv_last(attended_output)

        return out, last_state_list



    def _init_hidden(self, batch_size, height, width, device):

        init_states = []

        for i in range(self.num_layers):

            init_states.append([

                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),

                torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

            ])

        return init_states





class SSHDataset(Dataset):

    def __init__(self, sequences, targets, mask):

        self.sequences = torch.FloatTensor(sequences)

        self.targets = torch.FloatTensor(targets)

        self.mask = torch.FloatTensor(mask)



    def __len__(self):

        return len(self.sequences)



    def __getitem__(self, idx):

        return self.sequences[idx], self.targets[idx], self.mask[idx]





def load_and_preprocess_data(file_path, scaler=None, input_length=21, prediction_length=7, normalize=True,

                             training=False):

                                                    

    with h5py.File(file_path, 'r') as f:

        ssh_data = f['ssh'][:]

        ssh_data = np.transpose(ssh_data)



    print("Data shape after loading:", ssh_data.shape)



    mask = ~np.isnan(ssh_data)



    if normalize:

        if training:

            valid_data = ssh_data[mask]

            scaler = MinMaxScaler(feature_range=(-1, 1))

            scaler.fit(valid_data.reshape(-1, 1))

            save_scaler_params(scaler)

        elif scaler is None:

            scaler = load_scaler_params()



        ssh_data_reshaped = ssh_data.reshape(-1, 1)

        ssh_data_normalized = scaler.transform(ssh_data_reshaped)

        ssh_data = ssh_data_normalized.reshape(ssh_data.shape)



    ssh_data = np.nan_to_num(ssh_data, nan=0.0)



    return ssh_data, mask, scaler





def create_sliding_windows(ssh_data, mask, input_length, prediction_length):

                            

    total_time = ssh_data.shape[2]

    sequences = []

    targets = []

    masks = []



    for i in range(total_time - input_length - prediction_length + 1):

        seq = ssh_data[:, :, i:i + input_length]

        target = ssh_data[:, :, i + input_length:i + input_length + prediction_length]

        mask_seq = mask[:, :, i + input_length:i + input_length + prediction_length]



        sequences.append(seq)

        targets.append(target)

        masks.append(mask_seq)



    sequences = np.array(sequences)

    targets = np.array(targets)

    masks = np.array(masks)



    sequences = sequences.transpose(0, 3, 1, 2)

    sequences = np.expand_dims(sequences, axis=2)

    targets = targets.transpose(0, 3, 1, 2)

    masks = masks.transpose(0, 3, 1, 2)



    return sequences, targets, masks





def generate_time_cv_splits(total_time_steps, input_length, prediction_length, n_folds=5):

    














       

    min_time_needed = input_length + prediction_length

    usable_steps = total_time_steps - min_time_needed + 1

    val_size = usable_steps // n_folds



    splits = []

    for fold in range(n_folds):

                    

        val_end_step = usable_steps - fold * val_size

        val_start_step = val_end_step - val_size



        if val_start_step < 0:

            val_start_step = 0



                                 

        train_ranges = []

        if val_start_step > 0:

            train_ranges.append((0, val_start_step))

        if val_end_step < usable_steps:

            train_ranges.append((val_end_step, usable_steps))



        if len(train_ranges) == 0:

            print(f"Warning: fold {fold + 1} does not have enough training data; skipped")

            continue



        splits.append({

            'fold': fold + 1,

            'train_ranges': train_ranges,

            'val_range': (val_start_step, val_end_step),

        })



    return splits





def train_model(gpu_id=0, random_seed=42):

    set_random_seed(random_seed)

    device = select_gpu(gpu_id)



    if torch.cuda.is_available():

        torch.cuda.empty_cache()



          

    input_length = 21

    prediction_length = 7

    file_path = r"E:\Ocean modelling\data\ssh.mat"

    batch_size = 1

    num_epochs = 100

    base_lr = 0.00025

    weight_decay = 0.01

    patience = 10

    n_folds = 5



            

    base_results_dir = 'results_cv'

    os.makedirs(base_results_dir, exist_ok=True)



    print("Loading and preprocessing data...")

    ssh_data, mask_data, scaler = load_and_preprocess_data(

        file_path,

        input_length=input_length,

        prediction_length=prediction_length,

        normalize=True,

        training=True

    )



    total_time_steps = ssh_data.shape[2]

    print(f"Total time steps: {total_time_steps}")



                    

    cv_splits = generate_time_cv_splits(

        total_time_steps, input_length, prediction_length, n_folds=n_folds

    )



    print(f"\n{'=' * 60}")

    print(f"Time-series {n_folds}-fold cross-validation")

    print(f"Strategy: use 20% as validation data in each fold and all remaining data as training data")

    print(f"{'=' * 60}\n")



    all_fold_results = []



    start_fold = 1                  

    for split_info in cv_splits:

        fold = split_info['fold']

        if fold < start_fold:

            print(f"Skipping fold {fold} (completed)")

            continue

        train_ranges = split_info['train_ranges']

        val_range = split_info['val_range']



        print(f"\n{'=' * 60}")

        print(f"Fold {fold}/{n_folds}")

        print(f"{'=' * 60}")

        print(f"Training sliding-window indices: {train_ranges}")

        print(f"Validation sliding-window indices: [{val_range[0]}, {val_range[1]})")



                        

        val_time_start = val_range[0]

        val_time_end = val_range[1] + input_length + prediction_length - 1

        val_ssh = ssh_data[:, :, val_time_start:val_time_end]

        val_mask = mask_data[:, :, val_time_start:val_time_end]

        X_val, y_val, mask_val = create_sliding_windows(

            val_ssh, val_mask, input_length, prediction_length

        )



                                

        X_train_list, y_train_list, mask_train_list = [], [], []

        for tr_start, tr_end in train_ranges:

            tr_time_end = tr_end + input_length + prediction_length - 1

            tr_ssh = ssh_data[:, :, tr_start:tr_time_end]

            tr_mask = mask_data[:, :, tr_start:tr_time_end]

            X_t, y_t, m_t = create_sliding_windows(tr_ssh, tr_mask, input_length, prediction_length)

            X_train_list.append(X_t)

            y_train_list.append(y_t)

            mask_train_list.append(m_t)



        X_train = np.concatenate(X_train_list, axis=0)

        y_train = np.concatenate(y_train_list, axis=0)

        mask_train = np.concatenate(mask_train_list, axis=0)



        print(f"Training samples: {len(X_train)}")

        print(f"Validation samples: {len(X_val)}")

        print(f"Train/validation ratio: {len(X_train)}/{len(X_val)} = {len(X_train) / len(X_val):.2f}")



                   

        fold_dir = os.path.join(base_results_dir, f'fold_{fold}')

        os.makedirs(fold_dir, exist_ok=True)



                   

        train_dataset = SSHDataset(X_train, y_train, mask_train)

        val_dataset = SSHDataset(X_val, y_val, mask_val)



        g = torch.Generator()

        g.manual_seed(random_seed)



        train_loader = DataLoader(

            train_dataset, batch_size=batch_size, shuffle=True,

            worker_init_fn=set_worker_seed, generator=g

        )

        val_loader = DataLoader(

            val_dataset, batch_size=batch_size, worker_init_fn=set_worker_seed

        )



               

        set_random_seed(random_seed)                      

        model = AttentionConvLSTM(

            input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2,

            prediction_length=prediction_length, batch_first=True

        ).to(device)



                 

        total_params = sum(p.numel() for p in model.parameters())

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total model parameters: {total_params:,}, trainable parameters: {trainable_params:,}")



        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)



        scheduler = OneCycleLR(

            optimizer, max_lr=base_lr, epochs=num_epochs,

            steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy='cos'

        )



        criterion = RefinedTrendAwareLoss(alpha=0.4, beta=0.6, threshold=0.04, weight_multiplier=2.75).to(device)

        metrics_tracker = MetricsTracker()



        print(f"\nStarting training for fold {fold}...")

        print(f"Training set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples\n")



        best_val_loss = float('inf')

        no_improve_epochs = 0



        for epoch in range(num_epochs):

            epoch_start_time = time.time()

            model.train()

            train_loss = 0



            train_pbar = tqdm(train_loader, desc=f'Fold {fold} Epoch {epoch + 1}/{num_epochs} [Train]')

            for batch_x, batch_y, batch_mask in train_pbar:

                batch_x = batch_x.to(device)

                batch_y = batch_y.to(device)

                batch_mask = batch_mask.to(device)



                optimizer.zero_grad()

                output, _ = model(batch_x)

                loss = criterion(output, batch_y, batch_mask)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                scheduler.step()

                train_loss += loss.item()

                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})



            model.eval()

            val_loss = 0

            with torch.no_grad():

                val_pbar = tqdm(val_loader, desc=f'Fold {fold} Epoch {epoch + 1}/{num_epochs} [Val]')

                for batch_x, batch_y, batch_mask in val_pbar:

                    batch_x = batch_x.to(device)

                    batch_y = batch_y.to(device)

                    batch_mask = batch_mask.to(device)

                    output, _ = model(batch_x)

                    mask_sum = batch_mask.sum().clamp(min=1)

                    mse_loss = ((output - batch_y) ** 2 * batch_mask).sum() / mask_sum

                    val_loss += mse_loss.item()

                    val_pbar.set_postfix({'loss': f'{mse_loss.item():.4f}'})



            avg_train_loss = train_loss / len(train_loader)

            avg_val_loss = val_loss / len(val_loader)

            current_lr = scheduler.get_last_lr()[0]



            metrics_tracker.update(avg_train_loss, avg_val_loss, current_lr)

            metrics_tracker.plot_metrics(save_path=fold_dir)



            epoch_time = time.time() - epoch_start_time

            print(f'\nFold {fold} Epoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s')

            print(f'Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

            print(f'Learning Rate: {current_lr:.6f}')

            print(f'[Samples] training: {len(train_dataset)}, Validation set: {len(val_dataset)}')



            if torch.cuda.is_available():

                memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3

                memory_cached = torch.cuda.memory_reserved(device) / 1024 ** 3

                print(f'GPU Memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached')



                          

            epoch_checkpoint = {

                'fold': fold,

                'epoch': epoch + 1,

                'model_state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),

                'scheduler_state_dict': scheduler.state_dict(),

                'train_loss': avg_train_loss,

                'val_loss': avg_val_loss,

                'train_samples': len(train_dataset),

                'val_samples': len(val_dataset),

                'random_seed': random_seed,

            }

            torch.save(epoch_checkpoint, os.path.join(fold_dir, f'model_epoch_{epoch + 1}.pth'))



                      

            if avg_val_loss < best_val_loss:

                best_val_loss = avg_val_loss

                no_improve_epochs = 0

                torch.save(epoch_checkpoint, os.path.join(fold_dir, 'best_model.pth'))

                print(f'Best model saved (val_loss: {avg_val_loss:.4f})')

            else:

                no_improve_epochs += 1

                if no_improve_epochs >= patience:

                    print(f'\nEarly stopping: Fold {fold}at epoch {epoch + 1} stopped')

                    break



            if torch.cuda.is_available():

                torch.cuda.empty_cache()



        all_fold_results.append({

            'fold': fold,

            'best_val_loss': best_val_loss,

            'train_samples': len(train_dataset),

            'val_samples': len(val_dataset),

            'epochs_trained': epoch + 1,

        })



        print(f"\nFold {fold}completed, best validation loss: {best_val_loss:.4f}")



              

    print(f"\n{'=' * 60}")

    print("Cross-validation summary")

    print(f"{'=' * 60}")

    val_losses = []

    for r in all_fold_results:

        print(f"Fold {r['fold']}: best_val_loss={r['best_val_loss']:.4f}, "

              f"train={r['train_samples']}, val={r['val_samples']}, epochs={r['epochs_trained']}")

        val_losses.append(r['best_val_loss'])



    print(f"\nMean validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")



            

    with open(os.path.join(base_results_dir, 'cv_summary.json'), 'w') as f:

        json.dump(all_fold_results, f, indent=2)



    return all_fold_results





if __name__ == "__main__":

    gpu_id = 0

    random_seed = 42

    print(f"Starting training - GPU: {gpu_id}, random seed: {random_seed}")

    results = train_model(gpu_id=gpu_id, random_seed=random_seed)
