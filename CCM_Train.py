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
import json
import pandas as pd
import math
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import glob
import scipy.io
import random

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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
    print("Deterministic mode enabled, training results will be reproducible")


def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


class OptimizedGeostrophicVelocityModule(nn.Module):
    def __init__(self, lon_grid, lat_grid, dt=0.25, f0=8.4e-5, g=9.8):
        super(OptimizedGeostrophicVelocityModule, self).__init__()

        self.register_buffer('lon_grid', torch.FloatTensor(lon_grid))
        self.register_buffer('lat_grid', torch.FloatTensor(lat_grid))

        self.dt = dt
        self.f0 = f0
        self.g = g
        self.earth_radius = 6378100
        self.deg_to_rad = math.pi / 180

        cos_lat_grid = torch.cos(self.lat_grid * self.deg_to_rad)
        self.register_buffer('cos_lat_grid', cos_lat_grid)

        dx_factor = self.earth_radius * dt * self.deg_to_rad
        dy_factor = self.earth_radius * dt * self.deg_to_rad

        dx_grid = dx_factor * cos_lat_grid
        dy_grid = dy_factor * torch.ones_like(self.lat_grid)

        self.register_buffer('dx_grid', dx_grid)
        self.register_buffer('dy_grid', dy_grid)

        print(f"Optimized geostrophic velocity module initialized")
        print(f"Grid size: {lon_grid.shape}")
        print(f"Latitude range: {lat_grid.min():.2f}° to {lat_grid.max():.2f}°")
        print(f"Using vectorized computation, avoiding double loops")

    def compute_gradients_correct(self, ssh):
        batch_size, channels, height, width = ssh.shape

        if channels > 1:
            ssh = ssh.mean(dim=1, keepdim=True)

        deta_dx = torch.zeros_like(ssh)

        deta_dx[:, :, 1:-1, :] = (ssh[:, :, 2:, :] - ssh[:, :, :-2, :]) / (
                2 * self.dx_grid[1:-1, :].unsqueeze(0).unsqueeze(0)
        )

        deta_dx[:, :, 0:1, :] = (ssh[:, :, 1:2, :] - ssh[:, :, 0:1, :]) / (
            self.dx_grid[0:1, :].unsqueeze(0).unsqueeze(0)
        )

        deta_dx[:, :, -1:, :] = (ssh[:, :, -1:, :] - ssh[:, :, -2:-1, :]) / (
            self.dx_grid[-1:, :].unsqueeze(0).unsqueeze(0)
        )

        deta_dy = torch.zeros_like(ssh)

        deta_dy[:, :, :, 1:-1] = (ssh[:, :, :, 2:] - ssh[:, :, :, :-2]) / (
                2 * self.dy_grid[:, 1:-1].unsqueeze(0).unsqueeze(0)
        )

        deta_dy[:, :, :, 0:1] = (ssh[:, :, :, 1:2] - ssh[:, :, :, 0:1]) / (
            self.dy_grid[:, 0:1].unsqueeze(0).unsqueeze(0)
        )

        deta_dy[:, :, :, -1:] = (ssh[:, :, :, -1:] - ssh[:, :, :, -2:-1]) / (
            self.dy_grid[:, -1:].unsqueeze(0).unsqueeze(0)
        )

        return deta_dx, deta_dy

    def forward(self, ssh):
        deta_dx, deta_dy = self.compute_gradients_correct(ssh)

        u_g = -(self.g / self.f0) * deta_dy
        v_g = (self.g / self.f0) * deta_dx

        return u_g, v_g


class BatchTanhFilter(nn.Module):
    def __init__(self, filter_scales, dt=0.25):
        super(BatchTanhFilter, self).__init__()
        self.filter_scales = filter_scales
        self.dt = dt
        self.filters = nn.ModuleList()

        print(f"Creating {len(filter_scales)} tanh filters...")

        for i, l_scale in enumerate(filter_scales):
            kernel_size = int(l_scale * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1

            conv_layer = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                   padding=kernel_size // 2, bias=False)

            self._init_tanh_kernel(conv_layer, l_scale, kernel_size)

            for param in conv_layer.parameters():
                param.requires_grad = False

            self.filters.append(conv_layer)

            if i % 10 == 0:
                print(f"  Created filter {i + 1}/{len(filter_scales)}, scale={l_scale:.2f}, kernel_size={kernel_size}")

    def _init_tanh_kernel(self, conv_layer, l_scale, kernel_size):
        center = kernel_size // 2
        x, y = np.mgrid[0:kernel_size, 0:kernel_size]

        xx = x - center
        yy = y - center
        r = np.sqrt(xx ** 2 + yy ** 2)

        G1 = 0.5 - 0.5 * np.tanh(0.1 * (r - l_scale / 2))

        A = 1.0 / np.sum(G1)
        G = A * G1

        conv_layer.weight.data[0, 0] = torch.FloatTensor(G)

    def forward(self, x, scale_idx):
        return self.filters[scale_idx](x)


class BatchOptimizedSSHEnergyCascadeModule(nn.Module):
    def __init__(self, lon_grid, lat_grid, dt=0.25, rho0=1027.4, f0=8.4e-5, g=9.8):
        super(BatchOptimizedSSHEnergyCascadeModule, self).__init__()

        self.rho0 = rho0
        self.dt = dt

        self.geo_velocity = OptimizedGeostrophicVelocityModule(
            lon_grid, lat_grid, dt, f0, g
        )

        dtt = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        self.filter_scales = dtt / dt

        self.tanh_filters = BatchTanhFilter(self.filter_scales, dt)

        print(f"Batch energy cascade module initialized")
        print(f"Filter scales count: {len(self.filter_scales)}")
        print(f"Scale range: {self.filter_scales[0]:.1f} to {self.filter_scales[-1]:.1f} grid points")
        print(f"Grid resolution: {dt}°")
        print(f"Using batch computation, no loop processing")

    def batch_nanconv(self, input_tensor, filter_layer):
        nan_mask = torch.isnan(input_tensor)

        input_clean = torch.where(nan_mask, torch.zeros_like(input_tensor), input_tensor)

        weight_matrix = torch.where(nan_mask, torch.zeros_like(input_tensor), torch.ones_like(input_tensor))

        convolved = filter_layer(input_clean)
        weight_conv = filter_layer(weight_matrix)

        result = torch.where(weight_conv > 1e-10, convolved / weight_conv, torch.zeros_like(convolved))

        return result

    def compute_energy_cascade_correct(self, ssh):
        batch_size, channels, height, width = ssh.shape

        u_vel, v_vel = self.geo_velocity(ssh)

        uu_orig = u_vel * u_vel
        vv_orig = v_vel * v_vel
        uv_orig = u_vel * v_vel

        all_energy_flux = []

        max_scales_to_compute = min(20, len(self.filter_scales))

        for scale_idx in range(max_scales_to_compute):
            try:
                u_filtered = self.batch_nanconv(u_vel,
                                                lambda x: self.tanh_filters(x, scale_idx))
                v_filtered = self.batch_nanconv(v_vel,
                                                lambda x: self.tanh_filters(x, scale_idx))
                uu_filtered = self.batch_nanconv(uu_orig,
                                                 lambda x: self.tanh_filters(x, scale_idx))
                vv_filtered = self.batch_nanconv(vv_orig,
                                                 lambda x: self.tanh_filters(x, scale_idx))
                uv_filtered = self.batch_nanconv(uv_orig,
                                                 lambda x: self.tanh_filters(x, scale_idx))

                du_dx, du_dy = self.geo_velocity.compute_gradients_correct(u_filtered)
                dv_dx, dv_dy = self.geo_velocity.compute_gradients_correct(v_filtered)

                tau_xx = uu_filtered - u_filtered * u_filtered
                tau_yy = vv_filtered - v_filtered * v_filtered
                tau_xy = uv_filtered - u_filtered * v_filtered

                energy_flux = -self.rho0 * (
                        tau_xx * du_dx +
                        tau_yy * dv_dy +
                        tau_xy * (du_dy + dv_dx)
                )

                energy_flux = energy_flux.squeeze(1)
                all_energy_flux.append(energy_flux)

            except Exception as e:
                print(f"Warning: Scale {scale_idx} computation failed: {e}")
                all_energy_flux.append(torch.zeros(batch_size, height, width,
                                                   device=ssh.device, dtype=ssh.dtype))

        while len(all_energy_flux) < len(self.filter_scales):
            all_energy_flux.append(torch.zeros(batch_size, height, width,
                                               device=ssh.device, dtype=ssh.dtype))

        final_energy_flux = torch.stack(all_energy_flux, dim=1)

        return final_energy_flux, u_vel, v_vel

    def forward(self, ssh):
        return self.compute_energy_cascade_correct(ssh)


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

        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w)

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

            layer_output = torch.stack(output_inner, dim=1)

            attended_output = self.temporal_attention(layer_output)

            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        out = self.conv_last(attended_output)

        return out, last_state_list

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([
                torch.zeros(batch_size, self.hidden_dim, height, width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, height, width).cuda()
            ])
        return init_states


class SSHPhysicsGuidedConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, prediction_length=7,
                 batch_first=True, lon_grid=None, lat_grid=None, dx=1000, dy=1000, f0=1e-4, g=9.8, rho0=1027.4):
        super(SSHPhysicsGuidedConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.padding = kernel_size // 2
        self.prediction_length = prediction_length

        self.convlstm = AttentionConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            prediction_length=prediction_length,
            batch_first=batch_first
        )

        if lon_grid is not None and lat_grid is not None:
            self.energy_cascade = BatchOptimizedSSHEnergyCascadeModule(
                lon_grid=lon_grid,
                lat_grid=lat_grid,
                dt=0.25,
                rho0=rho0,
                f0=f0,
                g=g
            )
        else:
            print("Warning: No lon/lat data provided, energy cascade module not initialized")
            self.energy_cascade = None

        if self.energy_cascade is not None:
            n_scales = len(self.energy_cascade.filter_scales)
            self.fusion_layer = nn.Conv2d(hidden_dim + n_scales, hidden_dim, kernel_size=3, padding=1)
        else:
            self.fusion_layer = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.final_output = nn.Conv2d(hidden_dim, prediction_length, kernel_size=3, padding=1)

    def forward(self, x, hidden_state=None):
        b, t, c, h, w = x.size()

        convlstm_output, last_state_list = self.convlstm(x, hidden_state)

        last_h = last_state_list[-1][0]

        if self.energy_cascade is not None:
            try:
                last_ssh = x[:, -1, :, :, :]
                cascade_features, u_vel, v_vel = self.energy_cascade(last_ssh)

                combined_features = torch.cat([last_h, cascade_features], dim=1)

            except Exception as e:
                print(f"Energy cascade computation failed, using original features: {e}")

                n_scales = len(self.energy_cascade.filter_scales)
                zero_cascade = torch.zeros(last_h.shape[0], n_scales, last_h.shape[2], last_h.shape[3],
                                           device=last_h.device, dtype=last_h.dtype)
                combined_features = torch.cat([last_h, zero_cascade], dim=1)
        else:
            combined_features = last_h

        fused_features = F.leaky_relu(self.fusion_layer(combined_features))
        output = self.final_output(fused_features)

        return output, last_state_list


class BatchOptimizedSSHPhysicsConstrainedLoss(nn.Module):
    def __init__(self, lon_grid, lat_grid, alpha=0.4, beta=0.15, gamma=0.15, delta=0.3,
                 dt=0.25, f0=8.4e-5, g=9.8, rho0=1027.4):
        super(BatchOptimizedSSHPhysicsConstrainedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mse = nn.MSELoss(reduction='none')

        print("Initializing batch energy cascade loss function...")
        self.energy_cascade = BatchOptimizedSSHEnergyCascadeModule(
            lon_grid, lat_grid, dt, rho0, f0, g
        )
        print("Batch energy cascade loss function initialized")

    def forward(self, pred, target, mask, model=None):
        mse_loss = self.mse(pred, target)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        trend_loss = self.mse(pred_diff, target_diff)

        change_magnitude = torch.abs(target_diff)
        weight = 1.0 + 2.0 * (change_magnitude > 0.1).float()
        magnitude_loss = self.mse(pred_diff, target_diff) * weight

        trend_mask = mask[:, 1:] * mask[:, :-1]

        physics_loss = self.energy_cascade_constraint_batch(pred, target, mask)

        combined_loss = (
                self.alpha * (mse_loss * mask).sum() / mask.sum() +
                self.beta * (trend_loss * trend_mask).sum() / trend_mask.sum() +
                self.gamma * (magnitude_loss * trend_mask).sum() / trend_mask.sum() +
                self.delta * physics_loss
        )

        return combined_loss

    def energy_cascade_constraint_batch(self, pred_ssh, target_ssh, mask):
        batch_size, time_steps, height, width = pred_ssh.size()

        pred_reshaped = pred_ssh.view(-1, 1, height, width)
        target_reshaped = target_ssh.view(-1, 1, height, width)
        mask_reshaped = mask.view(-1, height, width)

        total_samples = batch_size * time_steps
        sample_ratio = 0.2
        num_samples = int(total_samples * sample_ratio)

        sample_indices = torch.randperm(total_samples, device=pred_ssh.device)[:num_samples]

        pred_sampled = pred_reshaped[sample_indices]
        target_sampled = target_reshaped[sample_indices]
        mask_sampled = mask_reshaped[sample_indices]

        try:
            pred_energy_flux, pred_u, pred_v = self.energy_cascade(pred_sampled)
            target_energy_flux, target_u, target_v = self.energy_cascade(target_sampled)

            n_scales_to_use = min(5, pred_energy_flux.shape[1])
            energy_flux_loss = self.mse(
                pred_energy_flux[:, :n_scales_to_use],
                target_energy_flux[:, :n_scales_to_use]
            )

            mask_expanded = mask_sampled.unsqueeze(1)

            energy_masked = (energy_flux_loss.mean(dim=1,
                                                   keepdim=True) * mask_expanded).sum() / mask_expanded.sum().clamp(
                min=1)

            physics_loss = energy_masked

        except Exception as e:
            print(f"Batch energy cascade computation failed: {e}")
            physics_loss = torch.tensor(0.0, device=pred_ssh.device)

        return physics_loss


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


def load_and_preprocess_data(file_path, scaler=None, input_length=21, prediction_length=7, normalize=True,
                             training=False):
    with h5py.File(file_path, 'r') as f:
        ssh_data = f['ssh'][:]
        ssh_data = np.transpose(ssh_data)

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


class SSHDataset(Dataset):
    def __init__(self, sequences, targets, mask):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.mask = torch.FloatTensor(mask)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.mask[idx]


def visualize_predictions_english(inputs, predictions, targets, masks, scaler, epoch, save_dir, num_samples=3):
    batch_size = inputs.shape[0]
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)

    vis_dir = f"{save_dir}/epoch_{epoch + 1:03d}_visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    if scaler:
        def inverse_normalize(data):
            shape = data.shape
            data_flat = data.reshape(-1, 1)
            data_inverse = scaler.inverse_transform(data_flat)
            return data_inverse.reshape(shape)

        predictions_original = inverse_normalize(predictions)
        targets_original = inverse_normalize(targets)
    else:
        predictions_original = predictions
        targets_original = targets

    for i, idx in enumerate(sample_indices):
        input_seq = inputs[idx, :, 0]
        prediction = predictions_original[idx]
        target = targets_original[idx]
        mask = masks[idx]

        input_steps = input_seq.shape[0]
        pred_steps = prediction.shape[0]

        height, width = input_seq.shape[1], input_seq.shape[2]
        fig, axes = plt.subplots(3, pred_steps, figsize=(pred_steps * 3, 9))

        if pred_steps == 1:
            axes = axes.reshape(3, 1)

        for t in range(pred_steps):
            ax = axes[0, t]
            im = ax.imshow(prediction[t], cmap='viridis')
            ax.set_title(f'Prediction t+{t + 1}', fontsize=10)
            if t == 0:
                ax.set_ylabel('Predicted SSH', fontsize=12)
            ax.axis('off')

            ax = axes[1, t]
            im = ax.imshow(target[t], cmap='viridis')
            ax.set_title(f'Ground Truth t+{t + 1}', fontsize=10)
            if t == 0:
                ax.set_ylabel('True SSH', fontsize=12)
            ax.axis('off')

            ax = axes[2, t]
            error = prediction[t] - target[t]
            error_masked = np.where(mask[t], error, np.nan)
            vmax = np.nanmax(np.abs(error_masked)) if not np.isnan(error_masked).all() else 1
            im = ax.imshow(error_masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Error t+{t + 1}', fontsize=10)
            if t == 0:
                ax.set_ylabel('Prediction Error', fontsize=12)
            ax.axis('off')

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='SSH (m)')

        fig.suptitle(f'Epoch {epoch + 1} - Sample {i + 1} Prediction Results', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plt.savefig(f"{vis_dir}/sample_{i + 1}.png", dpi=300, bbox_inches='tight')
        plt.close()


def load_lonlat_data():
    try:
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


def train_ssh_physics_guided_model_batch_optimized(experiment_name="ssh_physics_batch_optimized", random_seed=42):
    set_random_seed(random_seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{experiment_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading lon/lat data...")
    lon_grid, lat_grid = load_lonlat_data()

    if lon_grid is None or lat_grid is None:
        print("Cannot load lon/lat data, exiting training")
        return None, None, None

    model_config = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "random_seed": random_seed,
        "input_length": 21,
        "prediction_length": 7,
        "hidden_dim": 64,
        "num_layers": 1,
        "kernel_size": 3,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "max_epochs": 200,
        "patience": 15,
        "physics_params": {
            "dt": 0.25,
            "f0": 8.4e-5,
            "g": 9.8,
            "rho0": 1027.4
        },
        "loss_weights": {
            "alpha": 0.4,
            "beta": 0.15,
            "gamma": 0.15,
            "delta": 0.3
        },
        "optimized_physics": True,
        "batch_optimized": True,
        "vectorized_gradients": True,
        "matlab_equivalent": True,
        "deterministic": True
    }

    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(model_config, f, indent=4)

    print(f"Experiment results will be saved to: {save_dir}")
    print(f"Using random seed: {random_seed}")

    input_length = model_config["input_length"]
    prediction_length = model_config["prediction_length"]
    batch_size = model_config["batch_size"]
    base_lr = model_config["learning_rate"]
    weight_decay = model_config["weight_decay"]
    num_epochs = model_config["max_epochs"]
    patience = model_config["patience"]

    physics_params = model_config["physics_params"]
    dt = physics_params["dt"]
    f0 = physics_params["f0"]
    g = physics_params["g"]
    rho0 = physics_params["rho0"]

    loss_weights = model_config["loss_weights"]

    metrics_tracker = MetricsTracker()

    file_path = r"E:\data\ssh.mat"

    print("Loading and preprocessing data...")
    sequences, targets, masks, scaler = load_and_preprocess_data(
        file_path,
        input_length=input_length,
        prediction_length=prediction_length,
        normalize=True,
        training=True
    )

    save_scaler_params(scaler, f"{save_dir}/scaler_params.json")

    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        sequences, targets, masks, test_size=0.2, random_state=random_seed
    )

    train_dataset = SSHDataset(X_train, y_train, mask_train)
    test_dataset = SSHDataset(X_test, y_test, mask_test)

    rng = torch.Generator()
    rng.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=set_worker_seed,
        generator=rng
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=set_worker_seed
    )

    print("Initializing batch optimized model...")
    model = SSHPhysicsGuidedConvLSTM(
        input_dim=1,
        hidden_dim=model_config["hidden_dim"],
        kernel_size=model_config["kernel_size"],
        num_layers=model_config["num_layers"],
        prediction_length=prediction_length,
        batch_first=True,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        dx=1000,
        dy=1000,
        f0=f0,
        g=g,
        rho0=rho0
    ).cuda()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    print("Initializing batch optimized loss function...")
    criterion = BatchOptimizedSSHPhysicsConstrainedLoss(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        alpha=loss_weights["alpha"],
        beta=loss_weights["beta"],
        gamma=loss_weights["gamma"],
        delta=loss_weights["delta"],
        dt=dt,
        f0=f0,
        g=g,
        rho0=rho0
    ).cuda()

    print("Starting training...")
    print(f"Using batch optimized vectorized energy cascade implementation")
    print(f"Random seed: {random_seed}")
    print(f"Physics params: dt={dt}°, f0={f0}, rho0={rho0}")
    print(f"Loss weights: SSH={loss_weights['alpha']}, Physics={loss_weights['delta']}")
    print(f"Batch optimization: no loop processing")
    print(f"Deterministic mode enabled")

    best_val_loss = float('inf')
    no_improve_epochs = 0

    log_file = open(f"{save_dir}/training_log.txt", "w")
    log_file.write("Epoch,Train Loss,Val Loss,Learning Rate,Epoch Time(s)\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_x, batch_y, batch_mask in train_pbar:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_mask = batch_mask.cuda()

            optimizer.zero_grad()
            output, _ = model(batch_x)

            loss = criterion(output, batch_y, batch_mask, model)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        val_loss = 0

        visual_inputs = []
        visual_predictions = []
        visual_targets = []
        visual_masks = []

        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for i, (batch_x, batch_y, batch_mask) in enumerate(val_pbar):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_mask = batch_mask.cuda()

                output, _ = model(batch_x)
                loss = criterion(output, batch_y, batch_mask, model)
                val_loss += loss.item()

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                if i == 0:
                    visual_inputs.append(batch_x.cpu().numpy())
                    visual_predictions.append(output.cpu().numpy())
                    visual_targets.append(batch_y.cpu().numpy())
                    visual_masks.append(batch_mask.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        current_lr = scheduler.get_last_lr()[0]

        metrics_tracker.update(avg_train_loss, avg_val_loss, current_lr)
        metrics_tracker.plot_metrics(save_path=save_dir)

        visualize_predictions_english(
            visual_inputs[0],
            visual_predictions[0],
            visual_targets[0],
            visual_masks[0],
            scaler,
            epoch,
            save_dir,
            num_samples=3
        )

        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        log_file.write(f"{epoch + 1},{avg_train_loss:.6f},{avg_val_loss:.6f},{current_lr:.6f},{epoch_time:.2f}\n")
        log_file.flush()

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
                'config': model_config,
                'random_seed': random_seed
            }
            torch.save(checkpoint, f'{save_dir}/best_model.pth')
            print(f'Model saved, validation loss: {avg_val_loss:.4f}')
        else:
            no_improve_epochs += 1

            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': model_config,
                    'random_seed': random_seed
                }
                torch.save(checkpoint, f'{save_dir}/checkpoint_epoch_{epoch + 1}.pth')

            if no_improve_epochs >= patience:
                print(f'\nEarly stopping: {patience} epochs without improvement, stopped at epoch {epoch + 1}')
                break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {save_dir}")
    print(f"Random seed used: {random_seed}")
    log_file.close()

    return model, metrics_tracker, save_dir


if __name__ == "__main__":
    random_seed = 42

    print(f"Starting training - random seed: {random_seed}")
    model, metrics_tracker, save_dir = train_ssh_physics_guided_model_batch_optimized(
        experiment_name="ssh_physics_batch_opt",
        random_seed=random_seed
    )