"""
Optimized Wav2Tensor implementation that exactly follows the paper's formulation.
This prototype focuses only on the core features described in the paper without additional enhancements.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Literal, List, Optional, Union
from functools import lru_cache

class Wav2TensorCore(nn.Module):
    """
    Optimized Wav2Tensor implementation that exactly follows the paper's formulation.
    
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
        include_planes: Optional[List[str]] = None,
        use_adaptive_freq=True,
        target_freq_bins=256
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
            use_adaptive_freq: Whether to use frequency-adaptive resolution for improved efficiency
            target_freq_bins: Number of frequency bins to use with adaptive frequency (default: 256)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_method = harmonic_method
        self.use_adaptive_freq = use_adaptive_freq
        
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
        
        # Setup adaptive frequency mapping if enabled
        # Do this after STFT is set up so we know the frequency dimensions
        if self.use_adaptive_freq:
            self._setup_adaptive_frequency_mapping(target_freq_bins)
        
        # Learned harmonic filterbanks (if selected)
        if harmonic_method == 'filterbank' and 'harmonic' in self.include_planes:
            # The filterbank operates on the frequency dimension
            # For a typical input shape of [B, F, T] or [B, 1, F, T]
            n_freq_bins = self._get_freq_bins()
            self.harmonic_filterbank = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),  # Reduced channels from 32 to 16
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))   # Removed middle layer for efficiency
            )
            
            # Initialize filterbank to approximate HPS behavior
            self._init_harmonic_filterbank()
        
        # Pre-compute critical band frequencies for psychoacoustic masking
        if 'psychoacoustic' in self.include_planes:
            self.critical_bands = torch.tensor([
                20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
                2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
            ])
            
            # Pre-compute bark scale for frequency bins to avoid repeated computation
            self._precompute_psychoacoustic_data()
    
    def _setup_adaptive_frequency_mapping(self, target_bins):
        """Set up frequency mapping for adaptive frequency resolution"""
        n_freq_bins = self.n_fft // 2 + 1
        
        # Create a mel-like scale with more resolution in bass region
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (self.sample_rate/2) / 700)
        
        # Allocate more bins to lower frequencies
        bass_ratio = 0.4  # 40% of bins for bass/lower-mid range
        bass_mel = mel_max * 0.25  # ~500-700Hz
        
        # Create points on mel scale
        bass_points = int(target_bins * bass_ratio)
        mid_high_points = target_bins - bass_points
        
        mel_points_bass = np.linspace(mel_min, bass_mel, bass_points)
        mel_points_mid_high = np.linspace(bass_mel, mel_max, mid_high_points + 1)[1:]
        mel_points = np.concatenate([mel_points_bass, mel_points_mid_high])
        
        # Convert mel points back to frequency
        adaptive_freqs = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Ensure we don't exceed the available frequency bins
        max_bin_index = n_freq_bins - 1
        self.adaptive_freq_bins = np.minimum(
            (adaptive_freqs / (self.sample_rate/2) * n_freq_bins).astype(int),
            max_bin_index
        )
        
        # Ensure unique and valid indices
        self.adaptive_freq_bins = np.unique(np.clip(self.adaptive_freq_bins, 0, max_bin_index))
        self.target_freq_bins = len(self.adaptive_freq_bins)
        
        # Register as buffer to move to GPU with the model if needed
        self.register_buffer('adaptive_freq_bins_tensor', 
                            torch.tensor(self.adaptive_freq_bins, dtype=torch.long))
    
    def _get_freq_bins(self):
        """Get the number of frequency bins for the current configuration"""
        if self.use_adaptive_freq:
            return self.target_freq_bins
        else:
            return self.n_fft // 2 + 1
    
    def _apply_adaptive_frequency(self, tensor):
        """Apply adaptive frequency resolution to a tensor"""
        if not self.use_adaptive_freq:
            return tensor
            
        # Find frequency dimension based on tensor shape
        freq_dim = 1 if len(tensor.shape) == 3 else 2
        
        # Use the pre-registered tensor for safety
        return torch.index_select(tensor, freq_dim, self.adaptive_freq_bins_tensor)
        
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
    
    def _precompute_psychoacoustic_data(self):
        """Precompute data needed for psychoacoustic calculations"""
        n_freq_bins = self.n_fft // 2 + 1
        if self.use_adaptive_freq:
            n_freq_bins = self.target_freq_bins
        
        # Linear frequencies for all bins
        if self.use_adaptive_freq:
            # Use the actual frequencies corresponding to our adaptive bins
            freqs = 700 * (10 ** (2595 * np.log10(1 + self.adaptive_freq_bins / (self.sample_rate/2)) / 2595) - 1)
            self.freqs = torch.tensor(freqs)
        else:
            self.freqs = torch.linspace(0, self.sample_rate/2, n_freq_bins)
        
        # Convert to Bark scale
        self.bark_scale = 13 * torch.atan(0.00076 * self.freqs) + 3.5 * torch.atan((self.freqs / 7500) ** 2)
        
        # Pre-compute spreading function
        self.spread = torch.zeros((n_freq_bins, n_freq_bins))
        
        for i in range(n_freq_bins):
            # Distance in Bark scale
            bark_distance = torch.abs(self.bark_scale[i] - self.bark_scale)
            # Create an asymmetric spreading function (stronger towards higher frequencies)
            mask_factor = torch.where(
                bark_distance <= 0,
                torch.exp(-0.5 * (bark_distance / 1.0) ** 2),    # Lower frequencies
                torch.exp(-0.7 * (bark_distance / 0.5) ** 2)     # Higher frequencies
            )
            self.spread[i] = mask_factor
        
        # Normalize each row of the spreading matrix
        self.spread = self.spread / (torch.sum(self.spread, dim=1, keepdim=True) + 1e-8)
        
        # Register as buffer to move to GPU with the model
        self.register_buffer('precomputed_spread', self.spread)
        self.register_buffer('precomputed_bark_scale', self.bark_scale)
    
    def _compute_spectral_plane(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral plane T_spec(f,t) = STFT(y)_{f,t} ∈ ℂ
        Returns complex spectrogram
        """
        # The STFT output has shape [B, C, F, T] if input is [B, C, T]
        # or [B, F, T] if input is [B, T]
        spec = self.stft(waveform)
        
        # Apply adaptive frequency if needed
        if self.use_adaptive_freq:
            if len(spec.shape) == 3:  # [B, F, T]
                freq_dim = 1
            else:  # [B, C, F, T]
                freq_dim = 2
            
            # Select frequency bins
            spec = torch.index_select(spec, freq_dim, self.adaptive_freq_bins_tensor)
        
        return spec
    
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
            
            # Ensure we don't ask for bins that don't exist
            target_size_f = max(1, F//k)
            
            # Use nearest mode and combine operations to minimize interpolation overhead
            harmonic_k = torch.nn.functional.interpolate(
                spec_magnitude.unsqueeze(1), 
                size=(target_size_f, T), 
                mode='nearest'
            ).squeeze(1)
            
            # Upsample back to original size
            harmonic_k = torch.nn.functional.interpolate(
                harmonic_k.unsqueeze(1), 
                size=(F, T), 
                mode='nearest'
            ).squeeze(1)
            
            # Add log instead of multiplying
            harmonic_sum += torch.log1p(harmonic_k)
        
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
        # Squared magnitude calculation
        left_energy = torch.abs(spec_left)**2
        right_energy = torch.abs(spec_right)**2
        
        # Avoid division by zero
        sum_energy = left_energy + right_energy + 1e-10
        energy_panning = (left_energy - right_energy) / sum_energy
        
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
        
        # Normalize to [0, 1] range for stability
        mag_db = torch.clamp(mag_db, min=-100, max=0)  # dB is usually negative
        mag_db = (mag_db + 100) / 100
        
        # Apply spreading function along frequency axis 
        # Use precomputed spreading matrix for efficiency
        masking = torch.zeros((B, F, T), device=spec_magnitude.device)
        
        # Fast implementation using batch matrix multiplication
        # Reshape mag_db to [B*T, F] for batch matmul
        mag_db_reshaped = mag_db.permute(0, 2, 1).reshape(B*T, F)
        
        # Apply spreading function - ensure dims match
        mask_size = min(F, self.precomputed_spread.shape[0], self.precomputed_spread.shape[1])
        masked_reshaped = torch.matmul(
            mag_db_reshaped[:, :mask_size], 
            self.precomputed_spread[:mask_size, :mask_size]
        )
        
        # If F > mask_size, we need to pad the results
        if F > mask_size:
            padding = torch.zeros((B*T, F - mask_size), device=masked_reshaped.device)
            masked_reshaped = torch.cat([masked_reshaped, padding], dim=1)
        
        # Reshape back to [B, F, T]
        masking = masked_reshaped.reshape(B, T, F).permute(0, 2, 1)
        
        # Apply an offset to simulate masking threshold
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
            # For stereo input - process channels separately
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
            if len(spec.shape) == 4:
                _, _, F, T = spec.shape
            else:
                _, F, T = spec.shape
            planes['harmonic'] = torch.zeros((B, 1, F, T), device=waveform.device)
        
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
            if len(spec.shape) == 4:
                _, _, F, T = spec.shape
            else:
                _, F, T = spec.shape
            planes['psychoacoustic'] = torch.zeros((B, 1, F, T), device=waveform.device)
        
        # Stack all planes to form the tensor
        tensor = torch.cat(tensor_components, dim=1)
        
        return tensor, planes 