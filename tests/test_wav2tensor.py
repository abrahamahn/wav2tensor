"""
Basic tests for the Wav2Tensor implementation.
"""

import sys
import os
import pytest
import torch
import numpy as np

# Add parent directory to path to import wav2tensor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from wav2tensor import Wav2TensorCore

class TestWav2Tensor:
    """Test class for Wav2Tensor functionality."""
    
    @pytest.fixture
    def wav2tensor(self):
        """Create a Wav2Tensor instance for testing."""
        return Wav2TensorCore(sample_rate=16000, n_fft=1024, hop_length=256)
    
    @pytest.fixture
    def mono_waveform(self):
        """Create a simple mono test waveform."""
        # Create a 1-second sine wave at 440 Hz
        sample_rate = 16000
        t = torch.arange(0, 1.0, 1/sample_rate)
        waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [B, C, T]
        return waveform
    
    @pytest.fixture
    def stereo_waveform(self):
        """Create a simple stereo test waveform."""
        # Create a 1-second stereo signal with phase difference
        sample_rate = 16000
        t = torch.arange(0, 1.0, 1/sample_rate)
        left = torch.sin(2 * np.pi * 440 * t)
        right = torch.sin(2 * np.pi * 440 * (t + 0.001))  # Phase shifted
        waveform = torch.stack([left, right], dim=0).unsqueeze(0)  # [B, C, T]
        return waveform
    
    def test_tensor_shape(self, wav2tensor, mono_waveform):
        """Test that the output tensor has the correct shape."""
        tensor, _ = wav2tensor(mono_waveform)
        
        # Check basic properties
        assert tensor.dim() == 4, "Output tensor should have 4 dimensions"
        assert tensor.shape[0] == 1, "Batch dimension should be preserved"
        
        # The output should have planes for:
        # - spectral (real, imag) [2]
        # - harmonic [1]
        # - spatial (IPD, energy) [2]
        # - psychoacoustic [1]
        # Total: 6 planes
        # Some implementations might have more detailed breakdowns
        assert tensor.shape[1] >= 6, "Output should have at least 6 channels (planes)"
    
    def test_plane_types(self, wav2tensor, mono_waveform):
        """Test that all expected planes are present in the output."""
        _, planes = wav2tensor(mono_waveform)
        
        # Check that all expected planes are present
        expected_planes = ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        for plane in expected_planes:
            assert plane in planes, f"Missing plane: {plane}"
    
    def test_stereo_processing(self, wav2tensor, stereo_waveform):
        """Test processing of stereo signals."""
        _, planes = wav2tensor(stereo_waveform)
        
        # Spatial features should be non-zero for stereo
        spatial = planes['spatial']
        assert not torch.allclose(spatial, torch.zeros_like(spatial)), \
            "Spatial features should be non-zero for stereo signals"
    
    def test_mono_processing(self, wav2tensor, mono_waveform):
        """Test processing of mono signals."""
        _, planes = wav2tensor(mono_waveform)
        
        # Check basic properties of spectral plane
        spectral = planes['spectral']
        assert spectral.dtype == torch.complex64 or spectral.dtype == torch.complex128, \
            "Spectral plane should be complex-valued"
    
    def test_harmonic_structure(self, wav2tensor):
        """Test harmonic structure calculation with a multi-harmonic signal."""
        # Create a signal with strong harmonics
        sample_rate = 16000
        t = torch.arange(0, 1.0, 1/sample_rate)
        f0 = 440  # Hz
        
        # Create waveform with multiple harmonics
        waveform = torch.zeros_like(t)
        for i in range(1, 4):  # Add 3 harmonics
            waveform += (1/i) * torch.sin(2 * np.pi * i * f0 * t)
        
        # Normalize and reshape for processing
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [B, C, T]
        
        # Get harmonic plane
        _, planes = wav2tensor(waveform)
        harmonic = planes['harmonic']
        
        # The harmonic plane should have strong responses
        # We can't predict exact values, but it should not be all zeros
        assert not torch.allclose(harmonic, torch.zeros_like(harmonic)), \
            "Harmonic plane should have non-zero values for harmonic signal"
    
    def test_psychoacoustic_features(self, wav2tensor, mono_waveform):
        """Test that psychoacoustic features are calculated."""
        _, planes = wav2tensor(mono_waveform)
        psychoacoustic = planes['psychoacoustic']
        
        # Psychoacoustic features should be within a reasonable range (usually 0-1)
        assert torch.all(psychoacoustic <= 2), "Psychoacoustic features should be bounded"
        assert not torch.allclose(psychoacoustic, torch.zeros_like(psychoacoustic)), \
            "Psychoacoustic features should not be all zeros" 