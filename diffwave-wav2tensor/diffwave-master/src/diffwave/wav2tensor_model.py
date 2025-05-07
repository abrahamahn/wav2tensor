import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from wav2tensor.core import Wav2TensorCore

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Wav2TensorUpsampler(nn.Module):
    def __init__(self, n_planes):
        super().__init__()
        # Adjust input channels based on number of Wav2Tensor planes
        self.conv1 = ConvTranspose2d(n_planes, n_planes, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(n_planes, n_planes, [3, 32], stride=[1, 16], padding=[1, 8])
        
        # Add frequency dimension adjustment
        self.freq_conv = Conv1d(n_planes, n_planes, 1)
        
        # Add final projection to match residual channels
        self.output_proj = Conv1d(n_planes, 1, 1)

    def forward(self, x):
        # x shape: [B, n_planes, F, T]
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        
        # Adjust frequency dimension
        B, C, F, T = x.shape
        x = x.transpose(2, 3)  # [B, C, T, F]
        x = x.reshape(B * C * T, F, 1)  # Prepare for 1D conv
        x = self.freq_conv(x)
        x = x.view(B, C, T, -1).transpose(2, 3)  # Back to [B, C, F, T]
        
        # Project to single channel
        x = self.output_proj(x.transpose(1, 2)).transpose(1, 2)  # [B, 1, F, T]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(input_channels, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        
        conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWaveWav2Tensor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # Initialize Wav2Tensor encoder
        self.wav2tensor = Wav2TensorCore(
            sample_rate=params.sample_rate,
            n_fft=params.n_fft,
            hop_length=params.hop_samples,
            include_planes=['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        )
        
        # Count number of Wav2Tensor planes
        self.n_planes = 6  # 2 for spectral (real/imag) + 1 harmonic + 2 spatial + 1 psychoacoustic
        
        # Model components
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.wav2tensor_upsampler = Wav2TensorUpsampler(self.n_planes)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(1, params.residual_channels, 2**(i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step):
        # Get Wav2Tensor representation
        wav2tensor_repr, _ = self.wav2tensor(audio)
        
        # Process audio input
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        # Get diffusion embedding
        diffusion_step = self.diffusion_embedding(diffusion_step)
        
        # Process Wav2Tensor representation
        wav2tensor_cond = self.wav2tensor_upsampler(wav2tensor_repr)

        # Apply residual layers
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, wav2tensor_cond)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x 