"""
Simplified Wav2Tensor implementation that exactly follows the paper's formulation.
This prototype focuses only on the core features described in the paper without additional enhancements.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple

class Wav2TensorCore(nn.Module):
    """
    Simplified Wav2Tensor implementation that exactly follows the paper's formulation.
    
    Creates a structured multi-plane audio representation tensor T ∈ ℂ^(F×T'×D) with four planes:
    1. Spectral plane (d=1): T_spec(f,t) = STFT(y)_{f,t} ∈ ℂ
    2. Harmonic plane (d=2): T_harm(f,t) = ∏_{k=1}^{K} |STFT(y)_{f/k,t}|
    3. Spatial plane (d=3): T_spat(f,t) = { IPD(y_L, y_R)_{f,t}, (|y_L|^2 - |y_R|^2)/(|y_L|^2 + |y_R|^2) }
    4. Psychoacoustic plane (d=4): T_psy(f,t) = M(y)_{f,t}
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # STFT transform
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,
            window_fn=torch.hann_window,
            return_complex=True
        )
        
        # Harmonic Product Spectrum parameters
        self.K = 5  # Maximum number of harmonics to consider
        
        # Critical band frequencies (Bark scale) for psychoacoustic masking
        self.critical_bands = torch.tensor([
            20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
            2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
        ])

    def _compute_spectral_plane(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral plane T_spec(f,t) = STFT(y)_{f,t} ∈ ℂ
        Returns complex spectrogram
        """
        return self.stft(waveform)  # (B, C, F, T)
    
    def _compute_harmonic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane T_harm(f,t) = ∏_{k=1}^{K} |STFT(y)_{f/k,t}|
        Implements Harmonic Product Spectrum (HPS)
        """
        B, C, F, T = spec_magnitude.shape
        
        # Initialize with the original magnitude spectrum
        harmonic_product = spec_magnitude.clone()
        
        # Multiply by downsampled versions for each harmonic
        for k in range(2, self.K + 1):
            # For each harmonic k, we need the spectrum at f/k
            # We achieve this by downsampling in frequency dimension
            downsampled = torch.nn.functional.interpolate(
                spec_magnitude, 
                size=(F//k, T), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Upsample back to original size
            upsampled = torch.nn.functional.interpolate(
                downsampled, 
                size=(F, T), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Multiply with the accumulated product
            harmonic_product = harmonic_product * upsampled
        
        return harmonic_product
    
    def _compute_spatial_plane(self, spec_left: torch.Tensor, spec_right: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial plane T_spat(f,t) = { IPD(y_L, y_R)_{f,t}, (|y_L|^2 - |y_R|^2)/(|y_L|^2 + |y_R|^2) }
        Returns tensor with [IPD, energy_panning] channels
        """
        # Compute interaural phase difference (IPD)
        ipd = torch.angle(spec_left) - torch.angle(spec_right)
        
        # Compute energy panning
        left_energy = torch.abs(spec_left) ** 2
        right_energy = torch.abs(spec_right) ** 2
        energy_panning = (left_energy - right_energy) / (left_energy + right_energy + 1e-10)
        
        # Stack IPD and energy panning
        spatial_plane = torch.stack([ipd, energy_panning], dim=1)  # (B, 2, F, T)
        
        return spatial_plane
    
    def _compute_psychoacoustic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute psychoacoustic plane T_psy(f,t) = M(y)_{f,t}
        Implements a simplified masking threshold calculation
        """
        B, C, F, T = spec_magnitude.shape
        
        # Convert to dB scale
        mag_db = 20 * torch.log10(spec_magnitude + 1e-10)
        
        # Compute spreading function (simplified)
        spread = torch.exp(-0.5 * torch.arange(-10, 11, dtype=torch.float32) ** 2)
        spread = spread / spread.sum()
        
        # Apply spreading function along frequency axis
        masking = torch.zeros_like(mag_db)
        
        for b in range(B):
            for c in range(C):
                for t in range(T):
                    # Use 1D convolution as spreading function
                    masking[b, c, :, t] = torch.nn.functional.conv1d(
                        mag_db[b, c, :, t].view(1, 1, -1),
                        spread.to(mag_db.device).view(1, 1, -1),
                        padding=10
                    ).view(-1)
        
        # Apply threshold offset (simplification of MPEG psychoacoustic model)
        masking_threshold = masking - 20  # Offset by 20dB
        
        return masking_threshold
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Transform input waveform into structured tensor T ∈ ℂ^(F×T'×D)
        
        Args:
            waveform: (B, C, T) input waveform
                      For mono audio: (B, 1, T)
                      For stereo audio: (B, 2, T)
        
        Returns:
            tensor: (B, D, F, T') structured tensor with D=4 planes
            planes: dictionary with individual planes for easier access
        """
        B, C, T = waveform.shape
        
        # 1. Spectral Plane
        if C == 1:
            # For mono input
            spec = self._compute_spectral_plane(waveform)
            spec_left = spec
            spec_right = spec
        else:
            # For stereo input
            spec_left = self._compute_spectral_plane(waveform[:, 0:1])
            spec_right = self._compute_spectral_plane(waveform[:, 1:2])
            # Average for mono representation when needed
            spec = (spec_left + spec_right) / 2
        
        # Get magnitude for subsequent calculations
        spec_magnitude = torch.abs(spec)
        
        # 2. Harmonic Plane
        harmonic_plane = self._compute_harmonic_plane(spec_magnitude)
        
        # 3. Spatial Plane
        if C > 1:  # Only compute spatial plane for stereo
            spatial_plane = self._compute_spatial_plane(spec_left, spec_right)
        else:  # For mono, use zeros
            _, _, F, T = spec.shape
            spatial_plane = torch.zeros((B, 2, F, T), device=waveform.device)
        
        # 4. Psychoacoustic Plane
        psychoacoustic_plane = self._compute_psychoacoustic_plane(spec_magnitude)
        
        # Stack all planes to form the tensor
        # Reshape components to match dimensions: 
        # - spec is complex so we use real and imaginary parts (2 channels)
        # - spatial plane has IPD and energy panning (2 channels)
        # - psychoacoustic plane has masking threshold (1 channel)
        tensor = torch.cat([
            torch.real(spec).unsqueeze(1),  # Real part of spectral plane
            torch.imag(spec).unsqueeze(1),  # Imaginary part of spectral plane
            harmonic_plane,                 # Harmonic plane
            spatial_plane,                  # Spatial plane (2 channels)
            psychoacoustic_plane            # Psychoacoustic plane
        ], dim=1)
        
        # For easier access, also return individual planes
        planes = {
            'spectral': spec,                     # Complex spectrogram
            'harmonic': harmonic_plane,           # Harmonic structure
            'spatial': spatial_plane,             # Spatial cues
            'psychoacoustic': psychoacoustic_plane # Masking threshold
        }
        
        return tensor, planes 