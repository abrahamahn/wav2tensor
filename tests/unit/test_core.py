"""
Tests for core functionalities of Wav2Tensor.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wav2tensor import Wav2TensorCore


class TestWav2TensorCore:
    """Test class for Wav2Tensor core functionality."""

    def test_tensor_shape(self, wav2tensor_default: Wav2TensorCore, mono_waveform: torch.Tensor) -> None:
        """Test that the output tensor has the correct shape."""
        tensor, _ = wav2tensor_default(mono_waveform)

        assert tensor.dim() == 4, "Output tensor should have 4 dimensions"
        assert tensor.shape[0] == 1, "Batch dimension should be preserved"
        assert tensor.shape[1] >= 6, "Output should have at least 6 channels (planes)"

    def test_plane_types(self, wav2tensor_default: Wav2TensorCore, mono_waveform: torch.Tensor) -> None:
        """Test that all expected planes are present in the output."""
        _, planes = wav2tensor_default(mono_waveform)

        expected_planes = ["spectral", "harmonic", "spatial", "psychoacoustic"]
        for plane in expected_planes:
            assert plane in planes, f"Missing plane: {plane}"

    def test_stereo_processing(self, wav2tensor_default: Wav2TensorCore, stereo_waveform: torch.Tensor) -> None:
        """Test processing of stereo signals."""
        _, planes = wav2tensor_default(stereo_waveform)

        spatial = planes["spatial"]
        assert not torch.allclose(spatial, torch.zeros_like(spatial)), "Spatial features should be non-zero for stereo signals"

    def test_psychoacoustic_features(self, wav2tensor_default: Wav2TensorCore, mono_waveform: torch.Tensor) -> None:
        """Test that psychoacoustic features are calculated within expected bounds."""
        _, planes = wav2tensor_default(mono_waveform)
        psychoacoustic = planes["psychoacoustic"]

        assert torch.all(psychoacoustic <= 2), "Psychoacoustic features should be bounded"
        assert not torch.allclose(psychoacoustic, torch.zeros_like(psychoacoustic)), (
            "Psychoacoustic features should not be all zeros"
        )

    def test_adaptive_frequency(
        self,
        wav2tensor_with_adaptive_freq: Wav2TensorCore,
        mono_waveform: torch.Tensor,
    ) -> None:
        """Test that adaptive frequency reduces the frequency dimension."""
        tensor, _ = wav2tensor_with_adaptive_freq(mono_waveform)

        assert tensor.shape[2] <= 129, "Adaptive frequency should reduce frequency bins"
        assert 100 <= tensor.shape[2] <= 129, "Should have ~128 frequency bins with adaptive freq"

    def test_batch_processing(self, wav2tensor_default: Wav2TensorCore) -> None:
        """Test processing of batched input."""
        batch_size = 2
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1 / sample_rate)

        waveforms = []
        for i in range(batch_size):
            freq = 440 * (1 + i * 0.5)
            wave = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
            waveforms.append(wave)

        batch = torch.stack(waveforms, dim=0)
        tensor, _ = wav2tensor_default(batch)

        assert tensor.shape[0] == batch_size, "Batch dimension should be preserved"

    def test_device_compatibility(self, wav2tensor_default: Wav2TensorCore, mono_waveform: torch.Tensor) -> None:
        """Test compatibility with different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")

        device = torch.device("cuda")
        wav2tensor_default.to(device)
        mono_waveform_gpu = mono_waveform.to(device)

        tensor, _ = wav2tensor_default(mono_waveform_gpu)
        assert tensor.device.type == "cuda", "Output tensor should be on GPU"

    def test_parameter_validation(self) -> None:
        """Test that invalid parameters are properly validated."""
        with pytest.raises(ValueError):
            Wav2TensorCore(include_planes=["spectral", "invalid_plane"])

    def test_output_range(self, wav2tensor_default: Wav2TensorCore, mono_waveform: torch.Tensor) -> None:
        """Test that outputs are in a reasonable range."""
        _, planes = wav2tensor_default(mono_waveform)

        harmonic = planes["harmonic"]
        assert torch.all(harmonic >= 0), "Harmonic plane should be non-negative"

        spatial = planes["spatial"]
        ipd = spatial[:, 0] if spatial.dim() == 4 else spatial[0]
        assert torch.all(ipd >= -np.pi) and torch.all(ipd <= np.pi), "IPD should be between -pi and pi"

    def test_inverse_reconstruction(self, mono_waveform: torch.Tensor) -> None:
        """Test forward and inverse waveform reconstruction."""
        model = Wav2TensorCore(use_adaptive_freq=False)
        tensor, planes = model(mono_waveform)
        reconstructed = model.inverse(tensor, planes)
        assert reconstructed.shape[0] == mono_waveform.shape[0]
        assert reconstructed.shape[-1] > 0
