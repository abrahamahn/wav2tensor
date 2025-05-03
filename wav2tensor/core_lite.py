"""
Wav2TensorLite implementation with early fusion architecture and optimized computational efficiency.
This implementation focuses on maintaining core benefits while reducing computational overhead.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, List, Optional, Union, Dict

class Wav2TensorLite(nn.Module):
    """
    Wav2TensorLite: Early fusion architecture for structured multi-plane audio representation.
    
    Features:
    1. Log-compressed representation (8/16-bit compatible)
    2. Frequency-adaptive resolution (higher in bass region)
    3. Optional planes that can be toggled based on task
    4. Early fusion of planes rather than independent processing
    
    The representation preserves critical information while reducing computational overhead.
    """
    def __init__(
        self, 
        sample_rate=22050, 
        n_fft=1024, 
        hop_length=256,
        bit_depth=16,
        use_adaptive_freq=True,
        harmonic_method='hps',
        include_planes: Optional[List[str]] = None,
        fusion_method='concat'
    ):
        """
        Initialize the Wav2TensorLite encoder.
        
        Args:
            sample_rate: Audio sample rate (default: 22050Hz)
            n_fft: FFT size for STFT (default: 1024)
            hop_length: Hop length for STFT (default: 256)
            bit_depth: Bit depth for tensor quantization (8 or 16, default: 16)
            use_adaptive_freq: Whether to use frequency-adaptive resolution
            harmonic_method: Method for harmonic plane calculation
                             'hps': Harmonic Product Spectrum (original method)
                             'filterbank': Learned harmonic filterbanks
            include_planes: List of planes to include in the output tensor
                            Options: ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
                            Default: All planes are included
            fusion_method: Method to fuse planes ('concat', 'add', 'learned')
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_method = harmonic_method
        self.bit_depth = bit_depth
        self.use_adaptive_freq = use_adaptive_freq
        self.fusion_method = fusion_method
        
        # Calculate quantization scale based on bit depth
        if bit_depth == 8:
            self.quant_max = 255
        elif bit_depth == 16:
            self.quant_max = 65535
        else:
            raise ValueError(f"Bit depth must be 8 or 16, got {bit_depth}")
            
        # Default to including the most important planes
        if include_planes is None:
            # Based on ablation study, psychoacoustic is least important
            self.include_planes = ['spectral', 'harmonic', 'spatial']
        else:
            self.include_planes = include_planes
        
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
        
        # For adaptive frequency resolution
        if self.use_adaptive_freq:
            # Create frequency bins with more resolution in bass region
            # We'll use the mel scale as inspiration but customize it
            self._setup_adaptive_frequency_mapping()
        
        # Harmonic Product Spectrum parameters
        self.K = 3  # Reduced from 5 to 3 to focus on the strongest harmonics
        
        # Learned harmonic filterbanks (if selected)
        if harmonic_method == 'filterbank' and 'harmonic' in self.include_planes:
            n_freq_bins = self._get_freq_bins()
            self.harmonic_filterbank = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),  # Smaller network
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
            )
            self._init_harmonic_filterbank()
        
        # Set up fusion layer if using learned fusion
        if fusion_method == 'learned':
            # Calculate number of input channels for fusion
            n_channels = 2  # Real and imaginary for spectral
            if 'harmonic' in self.include_planes:
                n_channels += 1
            if 'spatial' in self.include_planes:
                n_channels += 2  # IPD and energy panning
            if 'psychoacoustic' in self.include_planes:
                n_channels += 1
                
            # Create a lightweight fusion network
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 4, kernel_size=1)  # Reduce to 4 channels
            )

    def _setup_adaptive_frequency_mapping(self):
        """Set up frequency mapping for adaptive frequency resolution"""
        n_freq_bins = self.n_fft // 2 + 1
        
        # Create a mel-like scale with more resolution in bass region
        # and less in high frequencies
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (self.sample_rate/2) / 700)
        
        # Create mel points with more points in lower frequencies
        # This gives us ~1/3 of bins focused on 0-500Hz
        target_bins = min(256, n_freq_bins)  # Cap at 256 frequency bins for efficiency
        
        # Allocate more bins to lower frequencies
        bass_ratio = 0.4  # 40% of bins for the bass/lower-mid range
        bass_mel = mel_max * 0.25  # Corresponds to ~500-700Hz depending on sample rate
        
        # Create points on mel scale
        bass_points = int(target_bins * bass_ratio)
        mid_high_points = target_bins - bass_points
        
        mel_points_bass = np.linspace(mel_min, bass_mel, bass_points)
        mel_points_mid_high = np.linspace(bass_mel, mel_max, mid_high_points + 1)[1:]
        mel_points = np.concatenate([mel_points_bass, mel_points_mid_high])
        
        # Convert mel points back to frequency
        self.adaptive_freqs = 700 * (10 ** (mel_points / 2595) - 1)
        self.adaptive_freq_bins = (self.adaptive_freqs / (self.sample_rate/2) * n_freq_bins).astype(int)
        
        # Ensure unique and valid indices
        self.adaptive_freq_bins = np.unique(np.clip(self.adaptive_freq_bins, 0, n_freq_bins - 1))
        self.target_freq_bins = len(self.adaptive_freq_bins)
    
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
        
        # Select frequency bins
        return torch.index_select(tensor, freq_dim, torch.tensor(self.adaptive_freq_bins, device=tensor.device))
    
    def _init_harmonic_filterbank(self):
        """Initialize the harmonic filterbank to approximate HPS behavior."""
        first_layer = self.harmonic_filterbank[0]
        with torch.no_grad():
            kernel_size = first_layer.kernel_size[0]
            mid_point = kernel_size // 2
            
            # Initialize 8 filters of the first layer
            for i in range(min(8, first_layer.out_channels)):
                # Create a filter focusing on harmonic patterns
                first_layer.weight[i, 0, :, 0] = 0
                first_layer.weight[i, 0, mid_point, 0] = 0.5  # f
                
                # For harmonics 2 through K
                for k in range(2, self.K + 1):
                    harmonic_offset = mid_point // k
                    if harmonic_offset > 0:
                        first_layer.weight[i, 0, mid_point - harmonic_offset, 0] = 0.3 / k
    
    def _compute_spectral_plane(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute spectral plane"""
        return self.stft(waveform)
    
    def _compute_harmonic_plane_hps(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """Compute harmonic plane using optimized HPS"""
        # Handle different input shapes
        if len(spec_magnitude.shape) == 4:
            B, C, F, T = spec_magnitude.shape
            # Average channels if needed
            if C > 1:
                spec_magnitude = spec_magnitude.mean(dim=1)  # [B, F, T]
            else:
                spec_magnitude = spec_magnitude.squeeze(1)  # [B, F, T]
        else:
            # Already [B, F, T]
            B, F, T = spec_magnitude.shape
        
        # Log domain computation for better stability
        harmonic_sum = torch.log1p(spec_magnitude.clone())
        
        # Add harmonics in log domain
        for k in range(2, self.K + 1):
            # For each harmonic k, we need the spectrum at f/k
            downsampled = torch.nn.functional.interpolate(
                spec_magnitude.unsqueeze(1), 
                size=(F//k, T), 
                mode='nearest'
            ).squeeze(1)
            
            # Upsample back to original size
            upsampled = torch.nn.functional.interpolate(
                downsampled.unsqueeze(1), 
                size=(F, T), 
                mode='nearest'
            ).squeeze(1)
            
            # Add log instead of multiplying
            harmonic_sum += torch.log1p(upsampled)
        
        # Convert back from log domain
        harmonic_product = torch.expm1(harmonic_sum)
        
        # Apply frequency-adaptive transform if needed
        if self.use_adaptive_freq:
            harmonic_product = self._apply_adaptive_frequency(harmonic_product)
        
        return harmonic_product.unsqueeze(1)  # Add channel dimension [B, 1, F, T]
    
    def _compute_harmonic_plane_filterbank(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """Compute harmonic plane using learned filterbanks"""
        # Handle different shapes like in the HPS method
        if len(spec_magnitude.shape) == 4:
            B, C, F, T = spec_magnitude.shape
            if C > 1:
                spec_magnitude = spec_magnitude.mean(dim=1)
            else:
                spec_magnitude = spec_magnitude.squeeze(1)
        else:
            B, F, T = spec_magnitude.shape
        
        # Apply frequency-adaptive transform if needed
        if self.use_adaptive_freq:
            spec_magnitude = self._apply_adaptive_frequency(spec_magnitude)
            
        # Add channel dimension for the conv2d layers
        x = spec_magnitude.unsqueeze(1)  # [B, 1, F, T]
        
        # Apply the harmonic filterbank
        harmonic_output = self.harmonic_filterbank(x)  # [B, 1, F, T]
        
        # Apply ReLU for non-negative output
        harmonic_output = torch.relu(harmonic_output)
        
        return harmonic_output
    
    def _compute_harmonic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """Compute harmonic plane using selected method"""
        if self.harmonic_method == 'filterbank':
            return self._compute_harmonic_plane_filterbank(spec_magnitude)
        else:  # Default to 'hps'
            return self._compute_harmonic_plane_hps(spec_magnitude)
    
    def _compute_spatial_plane(self, spec_left: torch.Tensor, spec_right: torch.Tensor) -> torch.Tensor:
        """Compute spatial plane with IPD and energy panning"""
        # Compute interaural phase difference (IPD)
        ipd = torch.angle(spec_left) - torch.angle(spec_right)
        
        # Compute energy panning
        left_energy = torch.abs(spec_left) ** 2
        right_energy = torch.abs(spec_right) ** 2
        energy_panning = (left_energy - right_energy) / (left_energy + right_energy + 1e-10)
        
        # Handle dimensions
        if len(ipd.shape) == 4:
            ipd = ipd.squeeze(1)
            energy_panning = energy_panning.squeeze(1)
        
        # Apply frequency-adaptive transform if needed
        if self.use_adaptive_freq:
            ipd = self._apply_adaptive_frequency(ipd)
            energy_panning = self._apply_adaptive_frequency(energy_panning)
            
        # Stack IPD and energy panning
        ipd = ipd.unsqueeze(1)  # [B, 1, F, T]
        energy_panning = energy_panning.unsqueeze(1)  # [B, 1, F, T]
        spatial_plane = torch.cat([ipd, energy_panning], dim=1)  # [B, 2, F, T]
        
        return spatial_plane
    
    def _compute_psychoacoustic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        """Simplified psychoacoustic plane computation"""
        # Handle shapes
        if len(spec_magnitude.shape) == 4:
            B, C, F, T = spec_magnitude.shape
            if C > 1:
                spec_magnitude = spec_magnitude.mean(dim=1)  # [B, F, T]
            else:
                spec_magnitude = spec_magnitude.squeeze(1)  # [B, F, T]
        else:
            B, F, T = spec_magnitude.shape
        
        # Convert to dB and normalize
        spec_magnitude = torch.clamp(spec_magnitude, min=1e-10)
        mag_db = 20 * torch.log10(spec_magnitude)
        mag_db = torch.clamp(mag_db, min=-100, max=0)
        mag_db = (mag_db + 100) / 100  # Normalize to [0, 1]
        
        # Simplified masking threshold simulation using a small kernel
        # for local masking effects (much faster than full computation)
        mag_padded = torch.nn.functional.pad(mag_db, (0, 0, 2, 2), mode='reflect')
        masking = torch.zeros_like(mag_db)
        
        # Asymmetric weighting factors (stronger masking toward higher frequencies)
        weights = torch.tensor([0.3, 0.5, 1.0, 0.7, 0.4], device=mag_db.device)
        weights = weights / weights.sum()
        
        # Apply a simple 5-point masking spread (much faster than full matrix mult)
        for i in range(5):
            masking += weights[i] * mag_padded[:, i:i+F, :]
        
        # Apply offset to simulate masking threshold
        masking_offset = 0.1  # Slightly reduced from original
        masking_threshold = masking - masking_offset
        masking_threshold = torch.clamp(masking_threshold, min=0.0, max=1.0)
        
        # Apply frequency-adaptive transform if needed
        if self.use_adaptive_freq:
            masking_threshold = self._apply_adaptive_frequency(masking_threshold)
        
        return masking_threshold.unsqueeze(1)  # [B, 1, F, T]
    
    def _normalize_for_quantization(self, tensor, min_val=None, max_val=None):
        """Normalize tensor to [0,1] range for quantization"""
        if min_val is None or max_val is None:
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            
        # Handle case where min==max (constant tensor)
        if min_val == max_val:
            return torch.zeros_like(tensor), min_val, max_val
            
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized, min_val, max_val
    
    def _quantize_tensor(self, tensor):
        """Apply log-compression and quantization to tensor"""
        # Log compression: log(1+x) maps [0,∞) -> [0,∞)
        tensor_log = torch.log1p(tensor * 100)  # Scale by 100 before log for better dynamic range
        
        # Normalize to [0,1]
        tensor_norm, t_min, t_max = self._normalize_for_quantization(tensor_log)
        
        # Quantize to bit depth
        tensor_quant = torch.round(tensor_norm * self.quant_max) / self.quant_max
        
        return tensor_quant, (t_min, t_max)
    
    def _fuse_planes(self, planes_dict):
        """Fuse planes using selected method"""
        components = []
        
        # Always include spectral components
        spec = planes_dict['spectral']
        spec_real = torch.real(spec)
        spec_imag = torch.imag(spec)
        
        # Ensure correct dimensions (B, C, F, T)
        if len(spec_real.shape) == 3:  # (B, F, T)
            spec_real = spec_real.unsqueeze(1)
            spec_imag = spec_imag.unsqueeze(1)
        
        components.extend([spec_real, spec_imag])
        
        # Add other planes if included
        if 'harmonic' in self.include_planes:
            components.append(planes_dict['harmonic'])
            
        if 'spatial' in self.include_planes:
            components.append(planes_dict['spatial'])
            
        if 'psychoacoustic' in self.include_planes:
            components.append(planes_dict['psychoacoustic'])
        
        # Apply fusion method
        if self.fusion_method == 'concat':
            # Simple concatenation along channel dimension
            return torch.cat(components, dim=1)
            
        elif self.fusion_method == 'add':
            # Sum all planes (with broadcasting)
            # First ensure all have same channel dimension by averaging
            adjusted_components = []
            for comp in components:
                if comp.shape[1] > 1:
                    adjusted_components.append(comp.mean(dim=1, keepdim=True))
                else:
                    adjusted_components.append(comp)
            
            # Sum along feature dimension
            return torch.stack(adjusted_components, dim=0).sum(dim=0)
            
        elif self.fusion_method == 'learned':
            # Apply learnable fusion layer
            fused = self.fusion_layer(torch.cat(components, dim=1))
            return fused
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Transform input waveform into fused tensor representation.
        
        Args:
            waveform: (B, C, T) input waveform
                      For mono audio: (B, 1, T)
                      For stereo audio: (B, 2, T)
        
        Returns:
            tensor: (B, D, F, T') fused tensor with selected planes
            metadata: dictionary with metadata (normalization factors, etc.)
        """
        B, C, T = waveform.shape
        
        # Dictionary to store computed planes and metadata
        planes = {}
        metadata = {
            "quant_params": {},
            "tensor_shape": {}
        }
        
        # 1. Spectral Plane (always computed)
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
        
        # Apply adaptive frequency if needed
        if self.use_adaptive_freq:
            spec = self._apply_adaptive_frequency(spec)
            if C > 1:
                spec_left = self._apply_adaptive_frequency(spec_left) 
                spec_right = self._apply_adaptive_frequency(spec_right)
        
        # Store in planes dictionary
        planes['spectral'] = spec
        
        # Get magnitude for subsequent calculations
        spec_magnitude = torch.abs(spec)
        
        # 2. Harmonic Plane (if included)
        if 'harmonic' in self.include_planes:
            harmonic_plane = self._compute_harmonic_plane(spec_magnitude)
            planes['harmonic'] = harmonic_plane
        
        # 3. Spatial Plane (if included and input is stereo)
        if 'spatial' in self.include_planes and C > 1:
            spatial_plane = self._compute_spatial_plane(spec_left, spec_right)
            planes['spatial'] = spatial_plane
        
        # 4. Psychoacoustic Plane (if included)
        if 'psychoacoustic' in self.include_planes:
            psychoacoustic_plane = self._compute_psychoacoustic_plane(spec_magnitude)
            planes['psychoacoustic'] = psychoacoustic_plane
        
        # Early fusion of planes
        fused_tensor = self._fuse_planes(planes)
        
        # Apply quantization if bit depth is specified
        if self.bit_depth:
            quantized_tensor, quant_params = self._quantize_tensor(fused_tensor)
            metadata["quant_params"] = quant_params
            output_tensor = quantized_tensor
        else:
            output_tensor = fused_tensor
        
        # Store shape information
        for name, plane in planes.items():
            metadata["tensor_shape"][name] = list(plane.shape)
        
        metadata["tensor_shape"]["fused"] = list(output_tensor.shape)
        metadata["sample_rate"] = self.sample_rate
        metadata["n_fft"] = self.n_fft
        metadata["hop_length"] = self.hop_length
        metadata["fusion_method"] = self.fusion_method
        metadata["bit_depth"] = self.bit_depth
        
        return output_tensor, metadata 