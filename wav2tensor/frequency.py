"""
Frequency mapping and psychoacoustic spreading utilities for Wav2Tensor.
"""

from __future__ import annotations

import numpy as np
import torch

VALID_PLANES: tuple[str, ...] = ("spectral", "harmonic", "spatial", "psychoacoustic")


def compute_adaptive_frequency_bins(
    sample_rate: int,
    n_fft: int,
    target_bins: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Compute mel-weighted adaptive frequency bin indices with higher bass resolution."""
    n_freq_bins = n_fft // 2 + 1
    mel_min = 0.0
    mel_max = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)

    bass_ratio = 0.4
    bass_mel = mel_max * 0.25
    bass_points = int(target_bins * bass_ratio)
    mid_high_points = target_bins - bass_points

    mel_points = np.concatenate(
        [
            np.linspace(mel_min, bass_mel, bass_points),
            np.linspace(bass_mel, mel_max, mid_high_points + 1)[1:],
        ]
    )
    adaptive_freqs = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    max_bin_index = n_freq_bins - 1
    adaptive_bins = np.unique(
        np.clip(
            (adaptive_freqs / (sample_rate / 2.0) * n_freq_bins).astype(int),
            0,
            max_bin_index,
        )
    )
    adaptive_bins_tensor = torch.tensor(adaptive_bins, dtype=torch.long)
    return adaptive_bins, adaptive_bins_tensor


def compute_psychoacoustic_spreading(
    sample_rate: int,
    n_fft: int,
    n_freq_bins: int,
    adaptive_freq_bins: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute Bark-scale spreading matrix and Bark frequency coordinates."""
    if adaptive_freq_bins is not None:
        bin_ratio = torch.tensor(adaptive_freq_bins, dtype=torch.float32) / (n_fft // 2)
        freqs = (sample_rate / 2.0) * bin_ratio
    else:
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_freq_bins)

    bark_scale = 13.0 * torch.atan(0.00076 * freqs) + 3.5 * torch.atan((freqs / 7500.0) ** 2)
    spread = torch.zeros((n_freq_bins, n_freq_bins))
    bark_diff_matrix = bark_scale.unsqueeze(0) - bark_scale.unsqueeze(1)

    lower_mask = bark_diff_matrix <= 0
    higher_mask = ~lower_mask

    spread[lower_mask] = torch.exp(-0.5 * (bark_diff_matrix[lower_mask] / 1.0) ** 2)
    spread[higher_mask] = torch.exp(-0.7 * (bark_diff_matrix[higher_mask] / 0.5) ** 2)
    spread = spread / (torch.sum(spread, dim=1, keepdim=True) + 1e-8)
    return spread, bark_scale


def interpolate_adaptive_spectrum(
    spec: torch.Tensor,
    n_fft: int,
    adaptive_freq_bins: np.ndarray,
    adaptive_freq_bins_tensor: torch.Tensor,
) -> torch.Tensor:
    """Interpolate an adaptive-frequency complex spectrogram back to full STFT bin resolution."""
    if spec.dim() == 4:
        spec = spec.squeeze(1)

    n_freq_bins = n_fft // 2 + 1
    spec_full = torch.zeros(
        (spec.shape[0], n_freq_bins, spec.shape[-1]),
        dtype=spec.dtype,
        device=spec.device,
    )
    bins_on_device = adaptive_freq_bins_tensor.to(spec.device)
    spec_full[:, bins_on_device, :] = spec

    for i in range(len(adaptive_freq_bins) - 1):
        start_bin = int(adaptive_freq_bins[i])
        end_bin = int(adaptive_freq_bins[i + 1])
        if end_bin - start_bin > 1:
            start_val = spec_full[:, start_bin : start_bin + 1, :]
            end_val = spec_full[:, end_bin : end_bin + 1, :]
            steps = torch.linspace(0, 1, end_bin - start_bin + 1, device=spec.device)[1:-1]
            for j, weight in enumerate(steps, 1):
                spec_full[:, start_bin + j, :] = start_val * (1.0 - weight) + end_val * weight

    return spec_full
