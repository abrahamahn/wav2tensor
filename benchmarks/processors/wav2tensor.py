"""
Wav2Tensor processor for benchmarking.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.processors.base import BaseProcessor
from wav2tensor.core import Wav2TensorCore
from wav2tensor.frequency import VALID_PLANES


class Wav2TensorProcessor(BaseProcessor):
    """Processor for Wav2Tensor representation with performance tracking."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        include_planes: list[str] | None = None,
        harmonic_method: str = "hps",
        use_adaptive_freq: bool = False,
        target_freq_bins: int = 256,
    ) -> None:
        super().__init__(sample_rate=sample_rate, name="wav2tensor")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_planes = include_planes or list(VALID_PLANES)
        self.harmonic_method = harmonic_method
        self.use_adaptive_freq = use_adaptive_freq
        self.target_freq_bins = target_freq_bins

        self.wav2tensor = Wav2TensorCore(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins,
        )

    def process(self, waveform: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process an audio waveform into a Wav2Tensor representation and measure performance."""
        with torch.no_grad():
            tensor, planes = self.wav2tensor(waveform)

        memory_bytes = tensor.element_size() * tensor.nelement()
        memory_kb = memory_bytes / 1024.0

        planes_info = {plane_name: list(plane_tensor.shape) for plane_name, plane_tensor in planes.items()}

        metrics: dict[str, Any] = {
            "memory_kb": memory_kb,
            "tensor_shape": list(tensor.shape),
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "harmonic_method": self.harmonic_method,
            "use_adaptive_freq": self.use_adaptive_freq,
            "target_freq_bins": self.target_freq_bins,
            "planes": planes_info,
        }
        return tensor, metrics
