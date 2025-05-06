"""
Tests for core functionalities of Wav2Tensor.
"""

import pytest
import torch
import numpy as np


class TestWav2TensorCore:
    """Test class for Wav2Tensor core functionality."""
    
    def test_tensor_shape(self, wav2tensor_default, mono_waveform):
        """Test that the output tensor has the correct shape."""
        tensor, _ = wav2tensor_default(mono_waveform)
        
        # Check basic properties
        assert tensor.dim() == 4, "Output tensor should have 4 dimensions"
        assert tensor.shape[0] == 1, "Batch dimension should be preserved"
        
        # Check channel dimension based on included planes
        assert tensor.shape[1] >= 5, "Output should have at least 5 channels (planes)"
    
    def test_plane_types(self, wav2tensor_default, mono_waveform):
        """Test that all expected planes are present in the output."""
        _, planes = wav2tensor_default(mono_waveform)
        
        # Check that all expected planes are present
        expected_planes = ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        for plane in expected_planes:
            assert plane in planes, f"Missing plane: {plane}"
    
    def test_adaptive_frequency(self, wav2tensor_with_adaptive_freq, mono_waveform):
        """Test that adaptive frequency reduces the frequency dimension."""
        tensor, planes = wav2tensor_with_adaptive_freq(mono_waveform)
        
        # Check that the frequency dimension is reduced
        assert tensor.shape[2] <= 129, "Adaptive frequency should reduce frequency bins"
        
        # With target_freq_bins=128, we should have around that many bins
        # (exact count may vary due to uniqueness constraint)
        assert 100 <= tensor.shape[2] <= 129, "Should have ~128 frequency bins with adaptive freq"
    
    def test_batch_processing(self, wav2tensor_default):
        """Test processing of batched input."""
        # Create a batch of 2 mono waveforms
        batch_size = 2
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        
        waveforms = []
        for i in range(batch_size):
            freq = 440 * (1 + i*0.5)  # Different frequencies
            wave = torch.sin(2 * np.pi * freq * t).unsqueeze(0)  # [C, T]
            waveforms.append(wave)
            
        batch = torch.stack(waveforms, dim=0)  # [B, C, T]
        
        # Process the batch
        tensor, planes = wav2tensor_default(batch)
        
        # Check that batch dimension is preserved
        assert tensor.shape[0] == batch_size, "Batch dimension should be preserved"
        
    def test_device_compatibility(self, wav2tensor_default, mono_waveform):
        """Test compatibility with different devices."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")
        
        # Move model and input to GPU
        device = torch.device("cuda")
        wav2tensor_default.to(device)
        mono_waveform_gpu = mono_waveform.to(device)
        
        # Process on GPU
        tensor, planes = wav2tensor_default(mono_waveform_gpu)
        
        # Check that output is on the same device
        assert tensor.device.type == "cuda", "Output tensor should be on GPU"
        
    def test_parameter_validation(self):
        """Test that invalid parameters are properly validated."""
        # Test invalid plane name
        with pytest.raises(ValueError):
            wav2tensor = Wav2TensorCore(include_planes=['spectral', 'invalid_plane'])
            
    def test_output_range(self, wav2tensor_default, mono_waveform):
        """Test that outputs are in a reasonable range."""
        _, planes = wav2tensor_default(mono_waveform)
        
        # Harmonic plane should be non-negative
        harmonic = planes['harmonic']
        assert torch.all(harmonic >= 0), "Harmonic plane should be non-negative"
        
        # Spatial plane values should be in a reasonable range
        spatial = planes['spatial']
        # IPD should be between -π and π
        if len(spatial.shape) == 4:
            ipd = spatial[:, 0]
        else:
            ipd = spatial[0]
        assert torch.all(ipd >= -np.pi) and torch.all(ipd <= np.pi), "IPD should be between -π and π" 