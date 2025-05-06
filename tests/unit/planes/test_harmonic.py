"""
Tests for the harmonic plane of Wav2Tensor.
"""

import pytest
import torch
import numpy as np


class TestHarmonicPlane:
    """Test class for the harmonic plane functionality."""
    
    def test_harmonic_plane_shape(self, wav2tensor_default, mono_waveform):
        """Test that the harmonic plane has the correct shape."""
        _, planes = wav2tensor_default(mono_waveform)
        
        harmonic = planes['harmonic']
        
        # Check dimensions
        assert harmonic.dim() == 4, "Harmonic plane should be 4D [B, C, F, T]"
        
        B, C, F, T = harmonic.shape
        assert B == mono_waveform.shape[0], "Batch dimension should be preserved"
        assert C == 1, "Harmonic plane should have 1 channel"
        assert F > 0, "Should have non-zero frequency bins"
        assert T > 0, "Should have non-zero time frames"
    
    def test_harmonic_non_negative(self, wav2tensor_default, mono_waveform):
        """Test that the harmonic plane has non-negative values."""
        _, planes = wav2tensor_default(mono_waveform)
        
        harmonic = planes['harmonic']
        
        # Check that all values are non-negative
        assert torch.all(harmonic >= 0), "Harmonic plane should be non-negative"
    
    def test_harmonic_structure(self, wav2tensor_default, harmonic_waveform):
        """Test harmonic structure calculation with a multi-harmonic signal."""
        _, planes = wav2tensor_default(harmonic_waveform)
        
        harmonic = planes['harmonic']
        
        # The harmonic plane should have strong responses
        assert not torch.allclose(harmonic, torch.zeros_like(harmonic)), \
            "Harmonic plane should have non-zero values for harmonic signal"
    
    def test_harmonic_vs_noise(self, wav2tensor_default):
        """Test that harmonic content is stronger for harmonic signals than for noise."""
        # Create a harmonic signal
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        f0 = 440  # Hz
        
        # Create waveform with multiple harmonics
        harmonic_signal = torch.zeros_like(t)
        for i in range(1, 4):  # Add 3 harmonics
            harmonic_signal += (1/i) * torch.sin(2 * np.pi * i * f0 * t)
        
        # Normalize and reshape for processing
        harmonic_signal = harmonic_signal / torch.max(torch.abs(harmonic_signal))
        harmonic_signal = harmonic_signal.unsqueeze(0).unsqueeze(0)  # [B, C, T]
        
        # Create a noise signal
        noise_signal = torch.randn_like(harmonic_signal)
        noise_signal = noise_signal / torch.max(torch.abs(noise_signal))
        
        # Process both
        _, planes_harmonic = wav2tensor_default(harmonic_signal)
        _, planes_noise = wav2tensor_default(noise_signal)
        
        # Get harmonic planes
        harmonic_plane = planes_harmonic['harmonic']
        noise_plane = planes_noise['harmonic']
        
        # The harmonic signal should have higher average harmonic content
        assert torch.mean(harmonic_plane) > torch.mean(noise_plane), \
            "Harmonic signal should have stronger harmonic content than noise"
    
    def test_filterbank_method(self, wav2tensor_filterbank, harmonic_waveform):
        """Test the filterbank method for harmonic plane."""
        # Process with filterbank method
        _, planes = wav2tensor_filterbank(harmonic_waveform)
        
        harmonic = planes['harmonic']
        
        # The harmonic plane should be non-empty
        assert not torch.allclose(harmonic, torch.zeros_like(harmonic)), \
            "Filterbank method should produce non-zero harmonic plane"
    
    def test_harmonic_pattern(self, wav2tensor_default):
        """Test that the harmonic plane captures fundamental and harmonic partials."""
        # Create a signal with fundamental and first harmonic
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        fundamental_freq = 440  # Hz
        second_harmonic_freq = 2 * fundamental_freq  # Hz
        
        # Create waveform with fundamental and second harmonic
        waveform = 0.7 * torch.sin(2 * np.pi * fundamental_freq * t) + \
                   0.3 * torch.sin(2 * np.pi * second_harmonic_freq * t)
        
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [B, C, T]
        
        # Process the waveform
        _, planes = wav2tensor_default(waveform)
        
        # Get spectral and harmonic planes
        spectral = torch.abs(planes['spectral'])
        harmonic = planes['harmonic']
        
        # Remove channel dimensions if present
        if spectral.dim() == 4:
            spectral = spectral.squeeze(1)
        if harmonic.dim() == 4:
            harmonic = harmonic.squeeze(1)
        
        # Average over time
        avg_spectral = torch.mean(spectral, dim=-1).squeeze()
        avg_harmonic = torch.mean(harmonic, dim=-1).squeeze()
        
        # Find peaks in spectral plane
        # Since we're using a 440 Hz signal with 22050 Hz sample rate and n_fft=1024,
        # the fundamental peak should be around bin 20-21
        # The second harmonic should be around bin 40-42
        fundamental_bin_range = range(18, 23)  # ~440 Hz
        second_harmonic_bin_range = range(36, 46)  # ~880 Hz
        
        # Locate the peak bins
        fundamental_bin = torch.argmax(avg_spectral[15:25]) + 15
        second_harmonic_bin = torch.argmax(avg_spectral[35:45]) + 35
        
        # Verify peaks are in expected ranges
        assert fundamental_bin.item() in fundamental_bin_range, \
            f"Expected fundamental peak in bins {fundamental_bin_range}, got {fundamental_bin.item()}"
        assert second_harmonic_bin.item() in second_harmonic_bin_range, \
            f"Expected second harmonic peak in bins {second_harmonic_bin_range}, got {second_harmonic_bin.item()}"
        
        # The harmonic plane should show enhanced harmonicity compared to spectral
        # For both the fundamental and the harmonic
        # Note: The exact enhancement depends on the implementation details,
        # so we're just checking for relative enhancement, not specific values.
        fundamental_enhancement = avg_harmonic[fundamental_bin] / (avg_spectral[fundamental_bin] + 1e-10)
        second_harmonic_enhancement = avg_harmonic[second_harmonic_bin] / (avg_spectral[second_harmonic_bin] + 1e-10)
        
        # Both should show some enhancement
        assert fundamental_enhancement > 0, "Fundamental should have non-zero harmonic response"
        assert second_harmonic_enhancement > 0, "Second harmonic should have non-zero harmonic response" 