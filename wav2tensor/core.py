"""
Simplified Wav2Tensor implementation that exactly follows the paper's formulation.
This prototype focuses only on the core features described in the paper without additional enhancements.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Literal, List, Optional, Union

class Wav2TensorCore(nn.Module):
    """
    Simplified Wav2Tensor implementation that exactly follows the paper's formulation.
    
    Creates a structured multi-plane audio representation tensor T ∈ ℂ^(F×T'×D) with four planes:
    1. Spectral plane (d=1): T_spec(f,t) = STFT(y)_{f,t} ∈ ℂ
    2. Harmonic plane (d=2): T_harm(f,t) = ∏_{k=1}^{K} |STFT(y)_{f/k,t}|
    3. Spatial plane (d=3): T_spat(f,t) = { IPD(y_L, y_R)_{f,t}, (|y_L|^2 - |y_R|^2)/(|y_L|^2 + |y_R|^2) }
    4. Psychoacoustic plane (d=4): T_psy(f,t) = M(y)_{f,t}
    """
    def __init__(
        self, 
        sample_rate=22050, 
        n_fft=1024, 
        hop_length=256, 
        harmonic_method='hps',
        include_planes: Optional[List[str]] = None
    ):
        """
        Initialize the Wav2Tensor encoder.
        
        Args:
            sample_rate: Audio sample rate (default: 22050Hz)
            n_fft: FFT size for STFT (default: 1024)
            hop_length: Hop length for STFT (default: 256)
            harmonic_method: Method for harmonic plane calculation
                             'hps': Harmonic Product Spectrum (original method)
                             'filterbank': Learned harmonic filterbanks
            include_planes: List of planes to include in the output tensor
                            Options: ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
                            Default: All planes are included
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_method = harmonic_method
        
        # Default to including all planes if not specified
        self.include_planes = include_planes or ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        
        # Validate plane selection
        for plane in self.include_planes:
            if plane not in ['spectral', 'harmonic', 'spatial', 'psychoacoustic']:
                raise ValueError(f"Invalid plane name: {plane}. Must be one of: 'spectral', 'harmonic', 'spatial', 'psychoacoustic'")
        
        # Always include spectral plane as it's fundamental
        if 'spectral' not in self.include_planes:
            self.include_planes.append('spectral')
            print("Warning: Spectral plane is mandatory and has been added to include_planes.")
        
        # STFT transform
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,  # Returns complex STFT
            window_fn=torch.hann_window
        )
        
        # Harmonic Product Spectrum parameters
        self.K = 3  # Reduced from 5 to 3 to focus on the strongest harmonics
        
        # Learned harmonic filterbanks (if selected)
        if harmonic_method == 'filterbank' and 'harmonic' in self.include_planes:
            # The filterbank operates on the frequency dimension
            # For a typical input shape of [B, F, T] or [B, 1, F, T]
            n_freq_bins = n_fft // 2 + 1
            self.harmonic_filterbank = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(5, 1), stride=1, padding=(2, 0)),  # Learn harmonic bands
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),  # Mid-level harmonic features
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))    # Compress to harmonic map
            )
            
            # Initialize filterbank to approximate HPS behavior
            self._init_harmonic_filterbank()
        
        # Critical band frequencies (Bark scale) for psychoacoustic masking
        self.critical_bands = torch.tensor([
            20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
            2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
        ])
    
    def _init_harmonic_filterbank(self):
        """Initialize the harmonic filterbank to approximate HPS behavior."""
        # Access the first layer of the harmonic filterbank
        first_layer = self.harmonic_filterbank[0]
        with torch.no_grad():
            # Initialize some filters to detect harmonic patterns
            # Create filters that look for energy at frequency f and its harmonics f/2, f/3
            kernel_size = first_layer.kernel_size[0]
            mid_point = kernel_size // 2
            
            # Simple initialization for a few filters (others will be randomly initialized)
            for i in range(min(8, first_layer.out_channels)):
                # Create a filter that looks for the ith harmonic relationship
                first_layer.weight[i, 0, :, 0] = 0
                # Add some positive weight at the center and at harmonically related positions
                first_layer.weight[i, 0, mid_point, 0] = 0.5  # f
                
                # For harmonics 2 through K
                for k in range(2, self.K + 1):
                    harmonic_offset = mid_point // k
                    if harmonic_offset > 0:
                        first_layer.weight[i, 0, mid_point - harmonic_offset, 0] = 0.3 / k
    
    def _compute_spectral_plane(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral plane T_spec(f,t) = STFT(y)_{f,t} ∈ ℂ
        Returns complex spectrogram
        """
        # The STFT output has shape [B, C, F, T] if input is [B, C, T]
        # or [B, F, T] if input is [B, T]
        return self.stft(waveform)
    
    def _compute_harmonic_plane_hps(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane using Harmonic Product Spectrum (HPS)
        T_harm(f,t) = ∏_{k=1}^{K} |STFT(y)_{f/k,t}|
        
        Args:
            spec_magnitude: Magnitude spectrogram [B, F, T] or [B, C, F, T]
            
        Returns:
            Harmonic plane tensor [B, 1, F, T]
        """
        # spec_magnitude shape could be [B, C, F, T] or [B, F, T]
        # Handle both cases
        if len(spec_magnitude.shape) == 4:
            B, C, F, T = spec_magnitude.shape
            # Collapse channel dimension by taking average if it exists
            if C > 1:
                spec_magnitude = spec_magnitude.mean(dim=1)  # Now [B, F, T]
            else:
                spec_magnitude = spec_magnitude.squeeze(1)  # Now [B, F, T]
        else:
            # Already [B, F, T]
            B, F, T = spec_magnitude.shape
        
        # Initialize with the log of the original magnitude spectrum to avoid vanishing values
        # Use log1p (log(1+x)) for numerical stability
        harmonic_sum = torch.log1p(spec_magnitude.clone())
        
        # Add log of downsampled versions for each harmonic instead of multiplying
        for k in range(2, self.K + 1):
            # For each harmonic k, we need the spectrum at f/k
            # We achieve this by downsampling in frequency dimension
            downsampled = torch.nn.functional.interpolate(
                spec_magnitude.unsqueeze(1), 
                size=(F//k, T), 
                mode='nearest'  # Changed from 'bilinear' to 'nearest' to preserve peak values
            ).squeeze(1)
            
            # Upsample back to original size
            upsampled = torch.nn.functional.interpolate(
                downsampled.unsqueeze(1), 
                size=(F, T), 
                mode='nearest'  # Changed from 'bilinear' to 'nearest' to preserve peak values
            ).squeeze(1)
            
            # Add log instead of multiplying
            harmonic_sum += torch.log1p(upsampled)
        
        # Convert back from log domain
        harmonic_product = torch.expm1(harmonic_sum)
        
        return harmonic_product.unsqueeze(1)  # Add channel dimension (B, 1, F, T)
    
    def _compute_harmonic_plane_filterbank(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane using learned harmonic filterbanks
        
        Args:
            spec_magnitude: Magnitude spectrogram [B, F, T] or [B, C, F, T]
            
        Returns:
            Harmonic plane tensor [B, 1, F, T]
        """
        # spec_magnitude shape could be [B, C, F, T] or [B, F, T]
        # Handle both cases
        if len(spec_magnitude.shape) == 4:
            B, C, F, T = spec_magnitude.shape
            # Collapse channel dimension by taking average if it exists
            if C > 1:
                spec_magnitude = spec_magnitude.mean(dim=1)  # Now [B, F, T]
            else:
                spec_magnitude = spec_magnitude.squeeze(1)  # Now [B, F, T]
        else:
            # Already [B, F, T]
            B, F, T = spec_magnitude.shape
        
        # Add channel dimension for the conv2d layers
        x = spec_magnitude.unsqueeze(1)  # [B, 1, F, T]
        
        # Apply the harmonic filterbank
        harmonic_output = self.harmonic_filterbank(x)  # [B, 1, F, T]
        
        # Apply ReLU to ensure non-negative outputs (like the HPS approach)
        harmonic_output = torch.relu(harmonic_output)
        
        return harmonic_output
    
    def _compute_harmonic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane using the selected method
        
        Args:
            spec_magnitude: Magnitude spectrogram
            
        Returns:
            Harmonic plane tensor [B, 1, F, T]
        """
        if self.harmonic_method == 'filterbank':
            return self._compute_harmonic_plane_filterbank(spec_magnitude)
        else:  # Default to 'hps'
            return self._compute_harmonic_plane_hps(spec_magnitude)
    
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
        
        # Ensure we have the right dimensionality
        if len(ipd.shape) == 4:
            ipd = ipd.squeeze(1)  # Remove channel dim if [B, 1, F, T] -> [B, F, T]
            energy_panning = energy_panning.squeeze(1)
            
        # Stack IPD and energy panning
        ipd = ipd.unsqueeze(1)  # Add channel dimension (B, 1, F, T)
        energy_panning = energy_panning.unsqueeze(1)  # Add channel dimension (B, 1, F, T)
        spatial_plane = torch.cat([ipd, energy_panning], dim=1)  # (B, 2, F, T)
        
        return spatial_plane
    
    def _compute_psychoacoustic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute psychoacoustic plane T_psy(f,t) = M(y)_{f,t}
        Implements a simplified masking threshold calculation
        """
        # spec_magnitude shape could be [B, C, F, T] or [B, F, T]
        # Handle both cases
        if len(spec_magnitude.shape) == 4:
            B, C, F, T = spec_magnitude.shape
            # Collapse channel dimension by taking average if it exists
            if C > 1:
                spec_magnitude = spec_magnitude.mean(dim=1)  # Now [B, F, T]
            else:
                spec_magnitude = spec_magnitude.squeeze(1)  # Now [B, F, T]
        else:
            # Already [B, F, T]
            B, F, T = spec_magnitude.shape
        
        # Convert to dB scale (limit to a reasonable range to avoid infinite values)
        spec_magnitude = torch.clamp(spec_magnitude, min=1e-10)
        mag_db = 20 * torch.log10(spec_magnitude)
        
        # Normalize to a reasonable range (0 to 1)
        mag_db = torch.clamp(mag_db, min=-100, max=0)  # dB is usually negative
        mag_db = (mag_db + 100) / 100  # Normalize to [0, 1] for stability
        
        # Convert linear frequency to Bark scale for better perceptual modeling
        # Simple approximation of Bark-scale mapping
        freqs = torch.linspace(0, self.sample_rate/2, F, device=spec_magnitude.device)
        bark_scale = 13 * torch.atan(0.00076 * freqs) + 3.5 * torch.atan((freqs / 7500) ** 2)
        
        # Create a spreading function based on psychoacoustic research
        # This spreading function is wider in lower frequencies, narrower in higher frequencies
        spread_width = 15  # Width of the spreading function in Bark units
        spread = torch.zeros((F, F), device=spec_magnitude.device)
        
        for i in range(F):
            # Distance in Bark scale
            bark_distance = torch.abs(bark_scale[i] - bark_scale)
            # Create an asymmetric spreading function (stronger towards higher frequencies)
            mask_factor = torch.where(
                bark_distance <= 0,
                torch.exp(-0.5 * (bark_distance / 1.0) ** 2),    # Lower frequencies (steeper slope)
                torch.exp(-0.7 * (bark_distance / 0.5) ** 2)     # Higher frequencies (more gradual slope)
            )
            spread[i] = mask_factor
        
        # Normalize each row of the spreading matrix
        spread = spread / (torch.sum(spread, dim=1, keepdim=True) + 1e-8)
        
        # Apply spreading function along frequency axis
        masking = torch.zeros_like(mag_db)
        
        for b in range(B):
            for t in range(T):
                # Apply the spreading function as a matrix multiplication
                masking[b, :, t] = torch.matmul(spread, mag_db[b, :, t])
        
        # Apply an offset to simulate masking threshold
        # Content below this threshold would typically be inaudible due to masking
        masking_offset = 0.12  # Represents masking threshold offset (~12 dB)
        masking_threshold = masking - masking_offset
        
        # Ensure the masking threshold is bounded between 0 and 1
        masking_threshold = torch.clamp(masking_threshold, min=0.0, max=1.0)
        
        return masking_threshold.unsqueeze(1)  # Add channel dimension (B, 1, F, T)
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Transform input waveform into structured tensor T ∈ ℂ^(F×T'×D)
        
        Args:
            waveform: (B, C, T) input waveform
                      For mono audio: (B, 1, T)
                      For stereo audio: (B, 2, T)
        
        Returns:
            tensor: (B, D, F, T') structured tensor with selected planes
            planes: dictionary with individual planes for easier access
        """
        B, C, T = waveform.shape
        
        # Initialize dictionary to store all computed planes
        planes = {}
        tensor_components = []
        
        # 1. Spectral Plane (always computed as it's needed for other planes)
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
        
        # Store spectral plane
        planes['spectral'] = spec
        
        # Get magnitude for subsequent calculations
        spec_magnitude = torch.abs(spec)
        
        # Prepare spectral components for tensor
        if len(spec.shape) == 3:
            # Add channel dimension
            spec_real = torch.real(spec).unsqueeze(1)
            spec_imag = torch.imag(spec).unsqueeze(1)
        else:
            # If already has channel dim, extract real and imaginary
            spec_real = torch.real(spec)
            spec_imag = torch.imag(spec)
            # If more than one channel, take average
            if spec_real.shape[1] > 1:
                spec_real = spec_real.mean(dim=1, keepdim=True)
                spec_imag = spec_imag.mean(dim=1, keepdim=True)
        
        # Add spectral components to tensor
        tensor_components.extend([spec_real, spec_imag])
        
        # 2. Harmonic Plane (if included)
        if 'harmonic' in self.include_planes:
            harmonic_plane = self._compute_harmonic_plane(spec_magnitude)
            planes['harmonic'] = harmonic_plane
            tensor_components.append(harmonic_plane)
        else:
            # Add empty placeholder for consistent dictionary access
            planes['harmonic'] = torch.zeros((B, 1, *spec_magnitude.shape[1:]), device=waveform.device)
        
        # 3. Spatial Plane (if included and input is stereo)
        if 'spatial' in self.include_planes and C > 1:
            spatial_plane = self._compute_spatial_plane(spec_left, spec_right)
            planes['spatial'] = spatial_plane
            tensor_components.append(spatial_plane)
        else:
            # Add empty placeholder with correct shape
            if len(spec.shape) == 4:
                _, _, F, T = spec.shape
            else:
                _, F, T = spec.shape
            planes['spatial'] = torch.zeros((B, 2, F, T), device=waveform.device)
            if 'spatial' in self.include_planes:
                tensor_components.append(planes['spatial'])
        
        # 4. Psychoacoustic Plane (if included)
        if 'psychoacoustic' in self.include_planes:
            psychoacoustic_plane = self._compute_psychoacoustic_plane(spec_magnitude)
            planes['psychoacoustic'] = psychoacoustic_plane
            tensor_components.append(psychoacoustic_plane)
        else:
            # Add empty placeholder for consistent dictionary access
            planes['psychoacoustic'] = torch.zeros((B, 1, *spec_magnitude.shape[1:]), device=waveform.device)
        
        # Stack all planes to form the tensor
        tensor = torch.cat(tensor_components, dim=1)
        
        return tensor, planes 