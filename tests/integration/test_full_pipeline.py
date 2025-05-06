"""
Integration tests for the full Wav2Tensor processing pipeline.
"""

import pytest
import torch
import numpy as np
import os


class TestFullPipeline:
    """Test class for the full processing pipeline."""
    
    def test_full_pipeline(self, wav2tensor_default, sample_audio_waveform):
        """Test that the entire pipeline works with real audio."""
        # Process the waveform
        tensor, planes = wav2tensor_default(sample_audio_waveform)
        
        # Check that all expected planes are present and have reasonable shapes
        expected_planes = ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        for plane in expected_planes:
            assert plane in planes, f"Missing plane: {plane}"
            
            # Check that the plane has a reasonable shape
            plane_tensor = planes[plane]
            assert plane_tensor.dim() >= 3, f"Plane {plane} should have at least 3 dimensions"
    
    def test_adaptive_pipeline(self, sample_audio_waveform):
        """Test that the entire pipeline works with adaptive frequency resolution."""
        from wav2tensor import Wav2TensorCore
        
        # Create model with adaptive frequency
        wav2tensor = Wav2TensorCore(
            use_adaptive_freq=True,
            target_freq_bins=128
        )
        
        # Process the waveform
        tensor, planes = wav2tensor(sample_audio_waveform)
        
        # Check that the frequency dimension is reduced
        spectral = planes['spectral']
        if spectral.dim() == 3:
            _, F, _ = spectral.shape
        else:
            _, _, F, _ = spectral.shape
            
        assert F <= 129, "Adaptive frequency should reduce frequency bins"
    
    def test_multi_configuration_consistency(self, sample_audio_waveform):
        """Test that different configurations produce consistent results for the same input."""
        from wav2tensor import Wav2TensorCore
        
        # Create models with different configurations
        base_model = Wav2TensorCore(sample_rate=22050, n_fft=1024, hop_length=256)
        filterbank_model = Wav2TensorCore(
            sample_rate=22050, 
            n_fft=1024, 
            hop_length=256,
            harmonic_method='filterbank'
        )
        
        # Process the waveform with both models
        _, base_planes = base_model(sample_audio_waveform)
        _, filterbank_planes = filterbank_model(sample_audio_waveform)
        
        # Spectral planes should be nearly identical between the two models
        # Convert to numpy for easier comparison
        base_spec = torch.abs(base_planes['spectral']).cpu().numpy()
        filterbank_spec = torch.abs(filterbank_planes['spectral']).cpu().numpy()
        
        # Check they're the same shape
        assert base_spec.shape == filterbank_spec.shape, "Spectral shapes should match between models"
        
        # Check they're nearly identical (allowing for small numerical differences)
        np.testing.assert_allclose(base_spec, filterbank_spec, rtol=1e-5, atol=1e-7)
    
    def test_stereo_mono_consistency(self):
        """Test that mono-to-stereo conversion gives consistent results."""
        from wav2tensor import Wav2TensorCore
        
        # Create a mono test waveform
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        mono_waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [B, C, T]
        
        # Create a stereo version with the same signal in both channels
        stereo_waveform = torch.cat([mono_waveform, mono_waveform], dim=1)  # [B, 2, T]
        
        # Initialize the model
        wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
        
        # Process both waveforms
        _, mono_planes = wav2tensor(mono_waveform)
        _, stereo_planes = wav2tensor(stereo_waveform)
        
        # Spectral planes should be similar
        mono_spec = torch.abs(mono_planes['spectral']).mean(dim=0) if mono_planes['spectral'].dim() == 4 else torch.abs(mono_planes['spectral'])
        stereo_spec = torch.abs(stereo_planes['spectral']).mean(dim=0) if stereo_planes['spectral'].dim() == 4 else torch.abs(stereo_planes['spectral'])
        
        # Allow for small differences due to averaging channels
        np.testing.assert_allclose(
            mono_spec.cpu().numpy(), 
            stereo_spec.cpu().numpy(), 
            rtol=1e-3, atol=1e-5
        )
        
        # For identical L/R channels, spatial plane should have near-zero IPD and panning
        spatial = stereo_planes['spatial']
        
        # IPD should be close to zero (first channel of spatial plane)
        assert torch.allclose(spatial[:, 0].abs(), torch.tensor(0.0), atol=1e-5), \
            "IPD should be close to zero for identical channels"
            
        # Energy panning should be close to zero (second channel of spatial plane)
        assert torch.allclose(spatial[:, 1].abs(), torch.tensor(0.0), atol=1e-5), \
            "Energy panning should be close to zero for identical channels" 