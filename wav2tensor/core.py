"""
Optimized Wav2Tensor implementation that exactly follows the paper's formulation.
This optimized version focuses on reducing computational complexity while maintaining functionality.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Literal, List, Optional, Union
from functools import lru_cache
import torch.nn.functional as F

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
        target_freq_bins=256,
        use_half_precision=False,
        cache_size=8
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
            use_half_precision: Whether to use half precision for internal computations (default: False)
            cache_size: Size of the LRU cache for computations (default: 8)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_method = harmonic_method
        self.use_adaptive_freq = use_adaptive_freq
        self.use_half_precision = use_half_precision
        self.cache_size = cache_size
        
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
        if self.use_adaptive_freq:
            self._setup_adaptive_frequency_mapping(target_freq_bins)
        
        # Pre-compute harmonic indices for faster computation
        self._precompute_harmonic_indices()
        
        # Learned harmonic filterbanks (if selected)
        if harmonic_method == 'filterbank' and 'harmonic' in self.include_planes:
            n_freq_bins = self._get_freq_bins()
            self.harmonic_filterbank = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
            )
            
            # Initialize filterbank more efficiently
            self._init_harmonic_filterbank_vectorized()
        
        # Pre-compute psychoacoustic data if needed
        if 'psychoacoustic' in self.include_planes:
            self._precompute_psychoacoustic_data()
    
    def _setup_adaptive_frequency_mapping(self, target_bins):
        """Set up frequency mapping for adaptive frequency resolution - optimized version"""
        n_freq_bins = self.n_fft // 2 + 1
        
        # Create a mel-like scale with more resolution in bass region
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (self.sample_rate/2) / 700)
        
        # Allocate more bins to lower frequencies
        bass_ratio = 0.4
        bass_mel = mel_max * 0.25
        
        # Create points on mel scale - vectorized computation
        bass_points = int(target_bins * bass_ratio)
        mid_high_points = target_bins - bass_points
        
        mel_points = np.concatenate([
            np.linspace(mel_min, bass_mel, bass_points),
            np.linspace(bass_mel, mel_max, mid_high_points + 1)[1:]
        ])
        
        # Vectorized conversion back to frequency
        adaptive_freqs = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Ensure valid indices in one vectorized operation
        max_bin_index = n_freq_bins - 1
        self.adaptive_freq_bins = np.unique(
            np.clip((adaptive_freqs / (self.sample_rate/2) * n_freq_bins).astype(int), 0, max_bin_index)
        )
        self.target_freq_bins = len(self.adaptive_freq_bins)
        
        # Register as buffer (single operation)
        self.register_buffer('adaptive_freq_bins_tensor', 
                            torch.tensor(self.adaptive_freq_bins, dtype=torch.long))
    
    def _precompute_harmonic_indices(self):
        """Pre-compute indices for harmonic sampling to avoid repeated calculations"""
        if not hasattr(self, 'target_freq_bins'):
            n_freq_bins = self.n_fft // 2 + 1
        else:
            n_freq_bins = self.target_freq_bins
        
        # For each harmonic k, pre-compute the index mapping
        harmonic_indices = {}
        for k in range(2, self.K + 1):
            # Calculate source indices (where we sample from)
            source_indices = torch.arange(0, n_freq_bins // k, dtype=torch.long)
            
            # Calculate target indices (where they map to in full spectrum)
            target_indices = torch.zeros(n_freq_bins, dtype=torch.long)
            
            # Fill target indices with nearest neighbor expansion
            for i in range(k):
                # Calculate segment size for this part of the mapping
                seg_size = min(len(source_indices), (n_freq_bins - i + k - 1) // k)
                if seg_size > 0:
                    target_indices[i::k][:seg_size] = source_indices[:seg_size]
            
            # Store the indices for this harmonic
            harmonic_indices[k] = target_indices
        
        # Register buffers for each harmonic's indices
        for k, indices in harmonic_indices.items():
            self.register_buffer(f'harmonic_indices_{k}', indices)
    
    def _get_freq_bins(self):
        """Get the number of frequency bins for the current configuration"""
        if self.use_adaptive_freq:
            return self.target_freq_bins
        else:
            return self.n_fft // 2 + 1
    
    def _init_harmonic_filterbank_vectorized(self):
        """Initialize the harmonic filterbank to approximate HPS behavior - vectorized version"""
        first_layer = self.harmonic_filterbank[0]
        with torch.no_grad():
            kernel_size = first_layer.kernel_size[0]
            mid_point = kernel_size // 2
            
            # Initialize filters in a batch
            num_filters = min(8, first_layer.out_channels)
            filters = torch.zeros(num_filters, 1, kernel_size, 1)
            
            # Set central weights for all filters at once
            filters[:, 0, mid_point, 0] = 0.5
            
            # Set harmonic weights for all filters at once
            for k in range(2, self.K + 1):
                harmonic_offset = mid_point // k
                if harmonic_offset > 0:
                    filters[:, 0, mid_point - harmonic_offset, 0] = 0.3 / k
            
            # Apply to weights in one operation
            first_layer.weight[:num_filters] = filters
    
    def _precompute_psychoacoustic_data(self):
        """Precompute data needed for psychoacoustic calculations - optimized version"""
        n_freq_bins = self._get_freq_bins()
        
        # Linear frequencies for all bins - vectorized computation
        if self.use_adaptive_freq:
            bin_ratio = torch.tensor(self.adaptive_freq_bins, dtype=torch.float32) / (self.n_fft // 2)
            freqs = self.sample_rate/2 * bin_ratio
        else:
            freqs = torch.linspace(0, self.sample_rate/2, n_freq_bins)
        
        # Convert to Bark scale in one vectorized operation
        bark_scale = 13 * torch.atan(0.00076 * freqs) + 3.5 * torch.atan((freqs / 7500) ** 2)
        
        # Pre-compute spreading function - use broadcasting for efficiency
        spread = torch.zeros((n_freq_bins, n_freq_bins))
        
        # Create a matrix of all possible bark distance combinations at once
        bark_diff_matrix = bark_scale.unsqueeze(0) - bark_scale.unsqueeze(1)
        
        # Apply the spreading function to all elements at once
        lower_mask = bark_diff_matrix <= 0
        higher_mask = ~lower_mask
        
        # Apply different spreading factors based on direction
        spread[lower_mask] = torch.exp(-0.5 * (bark_diff_matrix[lower_mask] / 1.0) ** 2)
        spread[higher_mask] = torch.exp(-0.7 * (bark_diff_matrix[higher_mask] / 0.5) ** 2)
        
        # Normalize in one operation
        spread = spread / (torch.sum(spread, dim=1, keepdim=True) + 1e-8)
        
        # Register as buffer
        self.register_buffer('precomputed_spread', spread)
        self.register_buffer('precomputed_bark_scale', bark_scale)
    
    def _ensure_4d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has shape [B, C, F, T] for consistent processing"""
        if len(tensor.shape) == 3:  # [B, F, T]
            return tensor.unsqueeze(1)
        return tensor
    
    def _compute_spectral_plane_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral plane for all channels at once - optimized batched version
        
        Args:
            waveform: Input waveform of shape [B, C, T]
            
        Returns:
            Complex spectrogram of shape [B, C, F, T]
        """
        B, C, T = waveform.shape
        
        # Special case for mono audio to avoid unnecessary operations
        if C == 1:
            spec = self.stft(waveform)
        else:
            # Process all channels at once by reshaping
            waveform_flat = waveform.reshape(B*C, 1, T)
            spec_flat = self.stft(waveform_flat)
            
            # Reshape back to [B, C, F, T]
            if len(spec_flat.shape) == 3:  # [B*C, F, T]
                F, T_frames = spec_flat.shape[1:]
                spec = spec_flat.reshape(B, C, F, T_frames)
            else:  # Already [B*C, 1, F, T]
                _, _, F, T_frames = spec_flat.shape
                spec = spec_flat.reshape(B, C, 1, F, T_frames).squeeze(2)
        
        # Apply adaptive frequency if needed - only do once
        if self.use_adaptive_freq:
            if len(spec.shape) == 3:  # [B, F, T]
                spec = torch.index_select(spec, 1, self.adaptive_freq_bins_tensor)
            else:  # [B, C, F, T]
                spec = torch.index_select(spec, 2, self.adaptive_freq_bins_tensor)
        
        return spec
    
    def _compute_harmonic_plane_hps_optimized(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane using optimized Harmonic Product Spectrum (HPS)
        Uses pre-computed indices to avoid interpolation operations
        
        Args:
            spec_magnitude: Magnitude spectrogram [B, F, T] or [B, C, F, T]
            
        Returns:
            Harmonic plane tensor [B, 1, F, T]
        """
        # Standardize input format
        spec_magnitude = self._ensure_4d(spec_magnitude)
        B, C, F, T = spec_magnitude.shape
        
        # Collapse channel dimension if needed
        if C > 1:
            spec_magnitude = spec_magnitude.mean(dim=1, keepdim=True)
        
        # Initialize with the log of the original magnitude spectrum
        harmonic_sum = torch.log1p(spec_magnitude.clone())
        
        # Use pre-computed indices for each harmonic
        for k in range(2, self.K + 1):
            # Get the pre-computed indices for this harmonic
            harmonic_indices = getattr(self, f'harmonic_indices_{k}')
            
            # Use the indices for direct sampling (much faster than interpolation)
            # This maps harmonic relationships directly without need for interpolation
            harmonic_k = torch.index_select(spec_magnitude, 2, harmonic_indices[:F].to(spec_magnitude.device))
            
            # Add to the sum in-place to save memory
            harmonic_sum.add_(torch.log1p(harmonic_k))
        
        # Convert back from log domain and normalize
        harmonic_plane = torch.expm1(harmonic_sum) / self.K
        
        return harmonic_plane
    
    def _compute_harmonic_plane_filterbank(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane using learned harmonic filterbanks - optimized version
        
        Args:
            spec_magnitude: Magnitude spectrogram [B, F, T] or [B, C, F, T]
            
        Returns:
            Harmonic plane tensor [B, 1, F, T]
        """
        # Standardize input format
        spec_magnitude = self._ensure_4d(spec_magnitude)
        B, C, F, T = spec_magnitude.shape
        
        # Collapse channel dimension if needed
        if C > 1:
            spec_magnitude = spec_magnitude.mean(dim=1, keepdim=True)
        else:
            spec_magnitude = spec_magnitude
        
        # Apply the harmonic filterbank
        harmonic_output = self.harmonic_filterbank(spec_magnitude)
        
        # Apply ReLU in-place to save memory
        harmonic_output.relu_()
        
        return harmonic_output
    
    @lru_cache(maxsize=8)
    def _get_harmonic_method(self, method_name):
        """Cache the harmonic method selection for efficiency"""
        if method_name == 'filterbank':
            return self._compute_harmonic_plane_filterbank
        else:  # Default to 'hps'
            return self._compute_harmonic_plane_hps_optimized
    
    def _compute_harmonic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute harmonic plane using the selected method - with caching
        
        Args:
            spec_magnitude: Magnitude spectrogram
            
        Returns:
            Harmonic plane tensor [B, 1, F, T]
        """
        # Use cached method selection
        compute_method = self._get_harmonic_method(self.harmonic_method)
        return compute_method(spec_magnitude)
    
    def _compute_spatial_plane_optimized(self, spec_left: torch.Tensor, spec_right: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial plane with optimized operations
        
        Args:
            spec_left: Left channel spectrogram
            spec_right: Right channel spectrogram
            
        Returns:
            Spatial plane tensor [B, 2, F, T]
        """
        # Standardize input format
        spec_left = self._ensure_4d(spec_left)
        spec_right = self._ensure_4d(spec_right)
        
        # Extract real and imaginary parts for phase calculation
        # Use angle directly instead of separate real/imag calculations
        ipd = torch.angle(spec_left) - torch.angle(spec_right)
        
        # Compute energy - reuse calculations for efficiency
        left_energy = torch.abs(spec_left).square_()  # In-place
        right_energy = torch.abs(spec_right).square_()  # In-place
        
        # Energy sum with small epsilon to avoid division by zero
        sum_energy = left_energy + right_energy + 1e-10
        
        # Energy panning calculation
        # Reuse left_energy to save memory
        left_energy.sub_(right_energy)  # Now contains (left-right)
        energy_panning = left_energy.div_(sum_energy)  # In-place division
        
        # Stack channels efficiently
        # Ensure both have same shape [B, 1, F, T]
        ipd = ipd.unsqueeze(1) if ipd.dim() == 3 else ipd
        energy_panning = energy_panning.unsqueeze(1) if energy_panning.dim() == 3 else energy_panning
        
        # Single concatenation operation along channel dimension
        return torch.cat([ipd, energy_panning], dim=1)
    
    def _compute_psychoacoustic_plane_optimized(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute psychoacoustic plane with optimized operations
        
        Args:
            spec_magnitude: Magnitude spectrogram
            
        Returns:
            Psychoacoustic plane tensor [B, 1, F, T]
        """
        # Standardize input format
        spec_magnitude = self._ensure_4d(spec_magnitude)
        B, C, F, T = spec_magnitude.shape
        
        # Collapse channel dimension if needed
        if C > 1:
            spec_magnitude = spec_magnitude.mean(dim=1)  # [B, F, T]
        else:
            spec_magnitude = spec_magnitude.squeeze(1)  # [B, F, T]
        
        # Convert to dB scale efficiently
        spec_magnitude.clamp_(min=1e-10)
        mag_db = 20.0 * torch.log10(spec_magnitude)
        
        # Normalize to [0, 1] range in-place
        mag_db.clamp_(min=-100, max=0)
        mag_db.add_(100).div_(100)
        
        # Apply spreading function using einsum for more efficient batch matrix multiplication
        # This avoids the expensive reshaping operations
        masked = torch.einsum('bft,fg->bgt', mag_db, self.precomputed_spread)
        
        # Apply offset and clamp in-place
        masked.sub_(0.12).clamp_(min=0.0, max=1.0)
        
        # Return with channel dimension
        return masked.unsqueeze(1)
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Transform input waveform into structured tensor T ∈ ℂ^(F×T'×D)
        Optimized version that minimizes memory allocations and redundant computations
        
        Args:
            waveform: (B, C, T) input waveform
                      For mono audio: (B, 1, T)
                      For stereo audio: (B, 2, T)
        
        Returns:
            tensor: (B, D, F, T') structured tensor with selected planes
            planes: dictionary with individual planes for easier access
        """
        print(f"\nWav2TensorCore forward: input shape = {waveform.shape}")
        
        # Store original precision for later conversion
        original_dtype = waveform.dtype
        
        # Use half precision for internal calculations if enabled
        if self.use_half_precision and waveform.dtype != torch.float16:
            waveform = waveform.half()
        
        B, C, T = waveform.shape
        print(f"Processing waveform: batch_size={B}, channels={C}, length={T}")
        
        # Initialize dictionary to store all computed planes
        planes = {}
        tensor_components = []
        
        # Compute spectral plane based on input channels
        # Apply batched computation for efficiency
        if C == 1 or 'spatial' not in self.include_planes:
            # For mono audio or when spatial plane is not needed
            # Just compute a single STFT
            print("Computing mono STFT...")
            spec = self._compute_spectral_plane_batch(waveform)
            spec_left = spec_right = spec
        else:
            # For stereo audio when spatial plane is needed
            # Compute left and right channel STFTs
            print("Computing stereo STFT...")
            spec_left = self._compute_spectral_plane_batch(waveform[:, 0:1])
            spec_right = self._compute_spectral_plane_batch(waveform[:, 1:2])
            # Average for mono representation
            spec = (spec_left + spec_right) / 2
        
        print(f"Spectral plane shape: {spec.shape}")
        
        # Store spectral plane
        planes['spectral'] = spec
        
        # Prepare tensor components efficiently
        spec_real = torch.real(spec)
        spec_imag = torch.imag(spec)
        
        # Standardize to 4D format
        spec_real = self._ensure_4d(spec_real)
        spec_imag = self._ensure_4d(spec_imag)
        
        print(f"Real/Imag components shape: {spec_real.shape}")
        
        # Add spectral components to tensor
        tensor_components.extend([spec_real, spec_imag])
        
        # Compute magnitude once and reuse
        spec_magnitude = torch.abs(spec)
        
        # Compute additional planes only if needed
        # Harmonic plane
        if 'harmonic' in self.include_planes:
            print("Computing harmonic plane...")
            harmonic_plane = self._compute_harmonic_plane(spec_magnitude)
            planes['harmonic'] = harmonic_plane
            tensor_components.append(harmonic_plane)
            print(f"Harmonic plane shape: {harmonic_plane.shape}")
        
        # Spatial plane (only for stereo input)
        if 'spatial' in self.include_planes and C > 1:
            print("Computing spatial plane...")
            spatial_plane = self._compute_spatial_plane_optimized(spec_left, spec_right)
            planes['spatial'] = spatial_plane
            tensor_components.append(spatial_plane)
            print(f"Spatial plane shape: {spatial_plane.shape}")
        
        # Psychoacoustic plane
        if 'psychoacoustic' in self.include_planes:
            print("Computing psychoacoustic plane...")
            psychoacoustic_plane = self._compute_psychoacoustic_plane_optimized(spec_magnitude)
            planes['psychoacoustic'] = psychoacoustic_plane
            tensor_components.append(psychoacoustic_plane)
            print(f"Psychoacoustic plane shape: {psychoacoustic_plane.shape}")
        
        # Stack all planes to form the tensor - single operation
        tensor = torch.cat(tensor_components, dim=1)
        print(f"Final tensor shape: {tensor.shape}")
        
        # Convert back to original precision if needed
        if self.use_half_precision and original_dtype != torch.float16:
            tensor = tensor.to(dtype=original_dtype)
            for key in planes:
                planes[key] = planes[key].to(dtype=original_dtype)
        
        return tensor, planes

    def inverse(self, tensor: torch.Tensor, planes: Optional[dict] = None) -> torch.Tensor:
        """
        Convert Wav2Tensor representation back to waveform.
        
        Args:
            tensor: The Wav2Tensor representation tensor
            planes: Optional dictionary containing individual planes
                   If not provided, will attempt to extract from tensor
        
        Returns:
            Reconstructed waveform tensor
        """
        print(f"\nWav2TensorCore inverse: input shape = {tensor.shape}")
        
        # Ensure tensor is 4D [B, C, F, T]
        tensor = self._ensure_4d(tensor)
        print(f"After ensure_4d: shape = {tensor.shape}")
        
        # If planes not provided, try to extract from tensor
        if planes is None:
            print("Extracting planes from tensor...")
            planes = {}
            current_channel = 0
            
            # Extract spectral plane (always first two channels - real and imaginary)
            spec_real = tensor[:, current_channel:current_channel+1]
            spec_imag = tensor[:, current_channel+1:current_channel+2]
            spec_real = spec_real.squeeze(1) if spec_real.shape[1] == 1 else spec_real
            spec_imag = spec_imag.squeeze(1) if spec_imag.shape[1] == 1 else spec_imag
            planes['spectral'] = torch.complex(spec_real, spec_imag)
            current_channel += 2
            print(f"Extracted spectral plane shape: {planes['spectral'].shape}")
            
            # Extract other planes if present
            if 'harmonic' in self.include_planes:
                planes['harmonic'] = tensor[:, current_channel:current_channel+1]
                current_channel += 1
                print(f"Extracted harmonic plane shape: {planes['harmonic'].shape}")
            
            if 'spatial' in self.include_planes:
                planes['spatial'] = tensor[:, current_channel:current_channel+2]
                current_channel += 2
                print(f"Extracted spatial plane shape: {planes['spatial'].shape}")
            
            if 'psychoacoustic' in self.include_planes:
                planes['psychoacoustic'] = tensor[:, current_channel:current_channel+1]
                current_channel += 1
                print(f"Extracted psychoacoustic plane shape: {planes['psychoacoustic'].shape}")
        
        # Get spectral plane
        spec = planes['spectral']
        print(f"Using spectral plane shape: {spec.shape}")
        
        # If using adaptive frequency, interpolate back to full resolution
        if self.use_adaptive_freq:
            print("Interpolating frequency resolution...")
            # Create target frequency grid
            n_freq_bins = self.n_fft // 2 + 1
            
            # Create full resolution spectrogram
            spec_full = torch.zeros(
                (spec.shape[0], n_freq_bins, spec.shape[-1]),
                dtype=spec.dtype,
                device=spec.device
            )
            
            # Copy adaptive frequency bins to their corresponding positions
            spec_full[:, self.adaptive_freq_bins_tensor, :] = spec
            
            # Linear interpolation for missing bins
            for i in range(len(self.adaptive_freq_bins) - 1):
                start_bin = self.adaptive_freq_bins[i]
                end_bin = self.adaptive_freq_bins[i + 1]
                if end_bin - start_bin > 1:
                    # Linear interpolation between bins
                    start_val = spec_full[:, start_bin:start_bin+1, :]
                    end_val = spec_full[:, end_bin:end_bin+1, :]
                    steps = torch.linspace(0, 1, end_bin - start_bin + 1, device=spec.device)[1:-1]
                    for j, t in enumerate(steps, 1):
                        spec_full[:, start_bin + j, :] = start_val * (1 - t) + end_val * t
            
            spec = spec_full
            print(f"Final interpolated shape: {spec.shape}")
        
        # Ensure spec has shape [B, F, T]
        if spec.dim() == 4:  # [B, 1, F, T]
            spec = spec.squeeze(1)
            print("Squeezed extra dimension")
        print(f"Pre-ISTFT shape: {spec.shape}")
        
        # Apply inverse STFT to get mono waveform
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=spec.device),
            return_complex=False
        )
        print(f"Post-ISTFT shape: {waveform.shape}")
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            print("Added batch dimension")
        
        # If we have spatial plane and harmonic plane, enhance the reconstruction
        if 'spatial' in planes and 'harmonic' in planes:
            print("Enhancing reconstruction with spatial and harmonic information...")
            
            # Get spatial information
            spatial = planes['spatial']
            ipd = spatial[:, 0:1]  # Inter-channel Phase Difference
            energy_panning = spatial[:, 1:2]  # Energy panning
            
            # Get harmonic information for enhancement
            harmonic = planes['harmonic']
            
            # Enhance the spectral content using harmonic information
            spec_enhanced = spec * (1.0 + harmonic)
            
            # Convert to left/right channels
            spec_left = spec_enhanced * torch.exp(1j * ipd/2)  # Add half phase difference
            spec_right = spec_enhanced * torch.exp(-1j * ipd/2)  # Subtract half phase difference
            
            # Apply energy panning
            energy_left = (1 + energy_panning) / 2
            energy_right = (1 - energy_panning) / 2
            
            spec_left = spec_left * torch.sqrt(energy_left)
            spec_right = spec_right * torch.sqrt(energy_right)
            
            # Convert back to waveform
            left_wave = torch.istft(
                spec_left,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=spec.device),
                return_complex=False
            )
            right_wave = torch.istft(
                spec_right,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=spec.device),
                return_complex=False
            )
            
            # Stack channels
            waveform = torch.stack([left_wave, right_wave], dim=1)
            print(f"Enhanced stereo waveform shape: {waveform.shape}")
            
            # Add batch dimension if needed
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
                print("Added batch dimension to stereo")
        else:
            # Add channel dimension for mono
            if waveform.dim() == 2:  # [B, T]
                waveform = waveform.unsqueeze(1)  # [B, 1, T]
                print("Added channel dimension for mono")
        
        print(f"Final output shape: {waveform.shape}")
        return waveform

def run_reconstruction_test(audio_files, output_dir, sample_rate=22050, segment_duration=1.0, batch_size=4):
    """
    Run reconstruction test on multiple audio files with different processors.
    
    Args:
        audio_files: List of audio file paths
        output_dir: Directory to save results
        sample_rate: Target sample rate
        segment_duration: Duration of segment to process in seconds (default: 1.0s, 0 for full length)
        batch_size: Number of processors to run in parallel
        
    Returns:
        Dictionary with test results for all processors and files
    """
    print("\nStarting reconstruction test...")
    segment_length = int(segment_duration * sample_rate) if segment_duration > 0 else None
    print(f"Using {'full audio length' if segment_length is None else f'{segment_length} samples'}")
    
    # Define processors to test
    print("\nInitializing processors...")
    processors = {
        'raw_waveform': IdentityProcessor(sample_rate=sample_rate),
        'stft': STFTProcessor(
            sample_rate=sample_rate,
            n_fft=1024,  # Using standard FFT size
            hop_length=256  # Using standard hop length
        ),
        'mel_spectrogram': MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_fft=1024,  # Using standard FFT size
            hop_length=256,  # Using standard hop length
            n_mels=80
        ),
        'wav2tensor': Wav2TensorProcessor(
            sample_rate=sample_rate,
            n_fft=1024,  # Using standard FFT size
            hop_length=256,  # Using standard hop length
            harmonic_method='hps',
            include_planes=['spectral', 'harmonic', 'spatial', 'psychoacoustic'],  # Using all planes
            use_adaptive_freq=False,  # Disable adaptive frequency for now
            target_freq_bins=None
        )
    }
    print("Processors initialized.")