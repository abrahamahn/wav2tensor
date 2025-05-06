"""
Tests for the spectral plane of Wav2Tensor.
"""

import pytest
import torch
import numpy as np


class TestSpectralPlane:
    """Test class for the spectral plane functionality."""
    
    def test_spectral_plane_shape(self, wav2tensor_default, mono_waveform):
        """Test that the spectral plane has the correct shape."""
        _, planes = wav2tensor_default(mono_waveform)
        
        spectral = planes['spectral']
        
        # Check dimensions
        assert spectral.dim() == 3 or spectral.dim() == 4, "Spectral plane should be 3D or 4D"
        
        # If 3D, it should be [B, F, T]
        if spectral.dim() == 3:
            B, F, T = spectral.shape
            assert B == mono_waveform.shape[0], "Batch dimension should be preserved"
            assert F > 0, "Should have non-zero frequency bins"
            assert T > 0, "Should have non-zero time frames"
        
        # If 4D, it should be [B, C, F, T]
        if spectral.dim() == 4:
            B, C, F, T = spectral.shape
            assert B == mono_waveform.shape[0], "Batch dimension should be preserved"
            assert C == 1, "Spectral plane should have 1 channel dimension"
            assert F > 0, "Should have non-zero frequency bins"
            assert T > 0, "Should have non-zero time frames"
    
    def test_spectral_plane_type(self, wav2tensor_default, mono_waveform):
        """Test that the spectral plane has complex values."""
        _, planes = wav2tensor_default(mono_waveform)
        
        spectral = planes['spectral']
        
        # Check that the spectral plane is complex-valued
        assert torch.is_complex(spectral), "Spectral plane should be complex-valued"
    
    def test_spectral_content_sine(self, wav2tensor_default, mono_waveform):
        """Test that a sine wave produces the expected spectral content."""
        _, planes = wav2tensor_default(mono_waveform)
        
        spectral = planes['spectral']
        
        # Get magnitude spectrum
        magnitude = torch.abs(spectral)
        
        # Reshape if needed for easier processing
        if magnitude.dim() == 4:
            magnitude = magnitude.squeeze(1)  # Remove channel dimension
        
        # A sine wave should have a peak at its frequency
        # Since we're using a 440 Hz signal with 22050 Hz sample rate and n_fft=1024,
        # the peak should be around bin 20-21
        # Get the index of the maximum value along the frequency axis
        max_bin = torch.argmax(torch.mean(magnitude, dim=-1), dim=-1)
        
        # Allow for some flexibility in the bin index due to spectral leakage
        expected_bin_range = range(18, 23)  # Approx 440 Hz bin ± 2 bins for leakage
        assert max_bin.item() in expected_bin_range, f"Expected peak in bins {expected_bin_range}, got {max_bin.item()}"
    
    def test_spectral_normalization(self, wav2tensor_default):
        """Test that scaling the input affects the spectral output proportionally."""
        # Create a reference sine wave
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        reference = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [B, C, T]
        
        # Create a scaled version
        scale_factor = 2.0
        scaled = reference * scale_factor
        
        # Process both
        _, planes_ref = wav2tensor_default(reference)
        _, planes_scaled = wav2tensor_default(scaled)
        
        # Get spectral planes
        spec_ref = planes_ref['spectral']
        spec_scaled = planes_scaled['spectral']
        
        # The magnitudes should differ by the scale factor
        mag_ref = torch.abs(spec_ref)
        mag_scaled = torch.abs(spec_scaled)
        
        # Allow for some numerical error
        ratio = mag_scaled / (mag_ref + 1e-10)  # Avoid division by zero
        assert torch.allclose(ratio, torch.tensor(scale_factor), rtol=0.1), \
            f"Expected magnitude ratio ~{scale_factor}, got mean: {torch.mean(ratio)}"
    
    def test_spectral_shift_invariance(self, wav2tensor_default):
        """Test that time-shifting the input only affects phase, not magnitude."""
        # Create a reference sine wave
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        reference = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [B, C, T]
        
        # Create a time-shifted version (shift by 10 samples)
        shift = 10
        shifted = torch.zeros_like(reference)
        shifted[..., shift:] = reference[..., :-shift]
        
        # Process both
        _, planes_ref = wav2tensor_default(reference)
        _, planes_shifted = wav2tensor_default(shifted)
        
        # Get spectral planes
        spec_ref = planes_ref['spectral']
        spec_shifted = planes_shifted['spectral']
        
        # The magnitudes should be similar
        mag_ref = torch.abs(spec_ref)
        mag_shifted = torch.abs(spec_shifted)
        
        # Compare magnitudes (focus on the main content region to avoid edge effects)
        center_frames = slice(10, -10)  # Skip some frames at edges
        assert torch.allclose(mag_ref[..., center_frames], mag_shifted[..., center_frames], rtol=0.2), \
            "Magnitude should be shift-invariant" 