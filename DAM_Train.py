import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import time
import warnings
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import json
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import random
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings('ignore')


def set_random_seed(seed=42):
    """
    Set all random seeds to ensure reproducibility of experiments

    Args:
        seed (int): Random seed value, default is 42
    """
    # Set Python built-in random module seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # If using CUDA, set CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set random seed for all GPUs

        # Set CUDA deterministic options
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set CUDA operation determinism (PyTorch 1.12+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)

    # Set environment variable to ensure determinism of certain operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to: {seed}")
    print("Deterministic mode enabled, training results will be reproducible")


def set_worker_seed(worker_id):
    """
    Set random seed for DataLoader worker processes
    This function is used for the worker_init_fn parameter of DataLoader

    Args:
        worker_id (int): Worker process ID
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# GPU selection function
def select_gpu(gpu_id=1):
    """
    Select specified GPU for training
    Args:
        gpu_id: GPU number, 0-3 corresponds to your four graphics cards
    Returns:
        device: torch device object
    """
    if not torch.cuda.is_available():
        print("CUDA unavailable, using CPU for training")
        return torch.device('cpu')

    # Check number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check if specified GPU is available
    if gpu_id >= num_gpus:
        print(f"Warning: Specified GPU {gpu_id} does not exist, using GPU 0")
        gpu_id = 0

    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    print(f"Device: {device}")

    return device


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, height, width)
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
        # x shape: (batch, seq_len, hidden_dim, height, width)
        b, t, c, h, w = x.size()
        # Reshape for attention computation
        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous()  # (b, h, w, t, c)
        x_reshaped = x_reshaped.view(-1, t, c)  # (b*h*w, t, c)

        # Calculate attention weights
        attn_weights = self.attention_layer(x_reshaped)  # (b*h*w, t, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply attention
        attended = torch.bmm(attn_weights.transpose(1, 2), x_reshaped)  # (b*h*w, 1, c)
        attended = attended.view(b, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)

        return attended


# Trend-aware loss function
class EnhancedTrendAwareLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25):
        super(EnhancedTrendAwareLoss, self).__init__()
        self.alpha = alpha  # MSE loss weight
        self.beta = beta  # Trend loss weight
        self.gamma = gamma  # Magnitude change loss weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask):
        # Basic MSE loss
        mse_loss = self.mse(pred, target)

        # Trend loss: capture change direction
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        trend_loss = self.mse(pred_diff, target_diff)

        # Magnitude change loss: pay special attention to dramatic changes
        change_magnitude = torch.abs(target_diff)
        weight = 1.0 + 2.0 * (change_magnitude > 0.1).float()  # Assign higher weight to large changes
        magnitude_loss = self.mse(pred_diff, target_diff) * weight

        # Adjust mask to match trend difference dimensions
        trend_mask = mask[:, 1:] * mask[:, :-1]

        # Combine losses
        combined_loss = (
                self.alpha * (mse_loss * mask).sum() / mask.sum() +
                self.beta * (trend_loss * trend_mask).sum() / trend_mask.sum() +
                self.gamma * (magnitude_loss * trend_mask).sum() / trend_mask.sum()
        )

        return combined_loss


class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def update(self, train_loss, val_loss, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

    def plot_metrics(self, save_path='results'):
        os.makedirs(save_path, exist_ok=True)
        epochs = range(1, len(self.train_losses) + 1)

        # 1. Main metrics plot (loss and learning rate)
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])

        # Loss curves (log scale)
        ax1 = plt.subplot(gs[0])
        ax1.semilogy(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.semilogy(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss (Log Scale)', fontsize=12)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss (log scale)')
        ax1.legend()
        ax1.grid(True)

        # Loss curves (linear scale)
        ax2 = plt.subplot(gs[1])
        ax2.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Training and Validation Loss (Linear Scale)', fontsize=12)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Learning rate curve
        ax3 = plt.subplot(gs[2])
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=12)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Loss comparison plot
        plt.figure(figsize=(10, 6))
        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
        plt.plot(epochs, loss_diff, 'purple', linewidth=2, label='Val Loss - Train Loss')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.fill_between(epochs, loss_diff, 0,
                         where=(loss_diff >= 0), color='red', alpha=0.1)
        plt.fill_between(epochs, loss_diff, 0,
                         where=(loss_diff <= 0), color='green', alpha=0.1)
        plt.title('Loss Difference (Validation - Training)', fontsize=12)
        plt.xlabel('Epochs')
        plt.ylabel('Loss Difference')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_path}/loss_difference.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Training dynamics plot
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
    """Save normalizer parameters"""
    params = {
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist(),
        'feature_range': scaler.feature_range
    }
    with open(save_path, 'w') as f:
        json.dump(params, f)


def load_scaler_params(load_path='scaler_params.json'):
    """Load normalizer parameters"""
    with open(load_path, 'r') as f:
        params = json.load(f)
    scaler = MinMaxScaler(feature_range=tuple(params['feature_range']))
    scaler.data_min_ = np.array(params['data_min'])
    scaler.data_max_ = np.array(params['data_max'])
    scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / (scaler.data_max_ - scaler.data_min_)
    scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
    return scaler


def load_and_preprocess_data(file_path, scaler=None, input_length=21, prediction_length=7, normalize=True,
                             training=False):
    """Load data and preprocess, supporting both training and prediction modes"""
    with h5py.File(file_path, 'r') as f:
        ssh_data = f['ssh'][:]
        ssh_data = np.transpose(ssh_data)

    mask = ~np.isnan(ssh_data)

    if normalize:
        if training:
            # Training mode: create new normalizer
            valid_data = ssh_data[mask]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(valid_data.reshape(-1, 1))
            # Save normalizer parameters
            save_scaler_params(scaler)
        elif scaler is None:
            # Prediction mode: load saved normalizer parameters
            scaler = load_scaler_params()

        ssh_data_reshaped = ssh_data.reshape(-1, 1)
        ssh_data_normalized = scaler.transform(ssh_data_reshaped)
        ssh_data = ssh_data_normalized.reshape(ssh_data.shape)

    ssh_data = np.nan_to_num(ssh_data, nan=0.0)

    # Build sequences
    sequences = []
    targets = []
    masks = []

    for i in range(len(ssh_data[0, 0]) - input_length - prediction_length + 1):
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

    return sequences, targets, masks, scaler


# Modified ConvLSTM cell
class AttentionConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super(AttentionConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.padding = nn.ReplicationPad2d(padding)
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=0)

        # Add spatial attention
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Apply spatial attention to input
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

        # Initialize temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim)

        # Initialize ConvLSTM cells with attention
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

        # Output projection
        self.conv_last = nn.Conv2d(hidden_dim, prediction_length, kernel_size=3, padding=1)

    def forward(self, x, hidden_state=None):
        b, t, _, h, w = x.size()
        device = x.device  # Get device of input tensor

        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w, device)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t_step in range(cur_layer_input.size(1)):
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[:, t_step, :, :, :],
                    (h, c)
                )
                output_inner.append(h)

            # Stack temporal outputs
            layer_output = torch.stack(output_inner, dim=1)  # (batch, seq_len, hidden_dim, height, width)

            # Apply temporal attention
            attended_output = self.temporal_attention(layer_output)

            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Use the temporally attended output for final prediction
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


# Dataset class
class SSHDataset(Dataset):
    def __init__(self, sequences, targets, mask):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.mask = torch.FloatTensor(mask)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.mask[idx]


def load_data(file_path, input_length=10, prediction_length=3, normalize=True):
    """Load data and perform normalization processing"""
    start_time = time.time()

    with h5py.File(file_path, 'r') as f:
        print("Available datasets:", list(f.keys()))
        ssh_data = f['ssh'][:]
        ssh_data = np.transpose(ssh_data)

    print("Data shape after loading:", ssh_data.shape)
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")

    mask = ~np.isnan(ssh_data)

    scaler = None
    if normalize:
        valid_data = ssh_data[mask]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(valid_data.reshape(-1, 1))

        ssh_data_reshaped = ssh_data.reshape(-1, 1)
        ssh_data_normalized = scaler.transform(ssh_data_reshaped)
        ssh_data = ssh_data_normalized.reshape(ssh_data.shape)

    ssh_data = np.nan_to_num(ssh_data, nan=0.0)

    sequences = []
    targets = []
    masks = []

    for i in range(len(ssh_data[0, 0]) - input_length - prediction_length):
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

    return sequences, targets, masks, scaler


def train_model(gpu_id=0, random_seed=42):
    """
    Model training function
    Args:
        gpu_id: Specify GPU number to use (0, 1, 2, 3)
        random_seed: Random seed to ensure experiment reproducibility
    """
    # ===== First set random seed =====
    set_random_seed(random_seed)

    # Select GPU
    device = select_gpu(gpu_id)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Training parameter settings
    input_length = 21
    prediction_length = 7
    file_path = r"E:\data\ssh.mat"
    batch_size = 16
    num_epochs = 200
    base_lr = 0.00025
    weight_decay = 0.01  # AdamW weight decay parameter
    patience = 15  # Early stopping patience value

    os.makedirs('results', exist_ok=True)
    metrics_tracker = MetricsTracker()

    print("Loading and preprocessing data...")
    sequences, targets, masks, scaler = load_and_preprocess_data(
        file_path,
        input_length=input_length,
        prediction_length=prediction_length,
        normalize=True,
        training=True
    )

    # Dataset split - use fixed random_state to ensure consistent splitting
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        sequences, targets, masks, test_size=0.2, random_state=random_seed)

    # Create data loaders - add deterministic parameters
    train_dataset = SSHDataset(X_train, y_train, mask_train)
    test_dataset = SSHDataset(X_test, y_test, mask_test)

    # Create random number generator for DataLoader
    g = torch.Generator()
    g.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=set_worker_seed,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        worker_init_fn=set_worker_seed
    )

    # Initialize model and move to specified device
    model = AttentionConvLSTM(
        input_dim=1,
        hidden_dim=64,
        kernel_size=3,
        num_layers=1,
        prediction_length=prediction_length,
        batch_first=True
    ).to(device)

    # Use AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Proportion of total training time for learning rate increase phase
        anneal_strategy='cos'  # Use cosine annealing strategy
    )

    # Use trend-aware loss function and move to specified device
    criterion = EnhancedTrendAwareLoss().to(device)

    print(f"Starting training with random seed: {random_seed}")
    print(f"Using GPU: {gpu_id}")
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0

        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_x, batch_y, batch_mask in train_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            output, _ = model(batch_x)

            loss = criterion(output, batch_y, batch_mask)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for batch_x, batch_y, batch_mask in val_pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)

                output, _ = model(batch_x)
                loss = criterion(output, batch_y, batch_mask)
                val_loss += loss.item()

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        current_lr = scheduler.get_last_lr()[0]

        # Update metrics tracker
        metrics_tracker.update(avg_train_loss, avg_val_loss, current_lr)
        metrics_tracker.plot_metrics()

        # Print training information
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # Display GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
            memory_cached = torch.cuda.memory_reserved(device) / 1024 ** 3
            print(f'GPU {gpu_id} Memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached')

        # Early stopping check and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'gpu_id': gpu_id,
                'random_seed': random_seed  # Save random seed information
            }
            torch.save(checkpoint, 'results/best_model.pth')
            print(f'Model saved at epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'\nEarly stopping after {epoch + 1} epochs')
                break

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Random seed used: {random_seed}")

    return model, metrics_tracker


if __name__ == "__main__":
    # Train using specified GPU and random seed
    # 0: First GPU, 1: Second GPU, 2: Third GPU, 3: Fourth GPU

    gpu_id = 1  # Modify this to specify which GPU to use
    random_seed = 42  # Set random seed, using the same seed can reproduce results

    print(f"Starting training - GPU: {gpu_id}, Random seed: {random_seed}")
    model, metrics_tracker = train_model(gpu_id=gpu_id, random_seed=random_seed)