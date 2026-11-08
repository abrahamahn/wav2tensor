"""
Multi-plane signal processing operators for Wav2Tensor (harmonic, spatial, psychoacoustic).
"""

from __future__ import annotations

import torch
from torch import nn


def ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [B, C, F, T] for consistent multi-plane processing."""
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    return tensor


def precompute_harmonic_indices(n_freq_bins: int, max_harmonic: int) -> dict[int, torch.Tensor]:
    """Precompute nearest-neighbor harmonic sampling indices for harmonics k = 2..K."""
    harmonic_indices: dict[int, torch.Tensor] = {}
    for k in range(2, max_harmonic + 1):
        source_indices = torch.arange(0, n_freq_bins // k, dtype=torch.long)
        target_indices = torch.zeros(n_freq_bins, dtype=torch.long)
        for offset in range(k):
            seg_size = min(len(source_indices), (n_freq_bins - offset + k - 1) // k)
            if seg_size > 0:
                target_indices[offset::k][:seg_size] = source_indices[:seg_size]
        harmonic_indices[k] = target_indices
    return harmonic_indices


def build_harmonic_filterbank(max_harmonic: int) -> nn.Sequential:
    """Create and initialize a 2D convolutional filterbank approximating HPS behavior."""
    filterbank = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),
        nn.ReLU(),
        nn.Conv2d(16, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)),
    )
    first_layer = filterbank[0]
    if isinstance(first_layer, nn.Conv2d):
        with torch.no_grad():
            kernel_size = first_layer.kernel_size[0]
            mid_point = kernel_size // 2
            num_filters = min(8, first_layer.out_channels)
            filters = torch.zeros(num_filters, 1, kernel_size, 1)
            filters[:, 0, mid_point, 0] = 0.5
            for k in range(2, max_harmonic + 1):
                harmonic_offset = mid_point // k
                if harmonic_offset > 0:
                    filters[:, 0, mid_point - harmonic_offset, 0] = 0.3 / k
            first_layer.weight[:num_filters] = filters
    return filterbank


def compute_harmonic_plane_hps(
    spec_magnitude: torch.Tensor,
    max_harmonic: int,
    harmonic_indices: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Compute harmonic plane using log-domain Harmonic Product Spectrum (HPS)."""
    spec_magnitude = ensure_4d(spec_magnitude)
    _, channels, freq_bins, _ = spec_magnitude.shape
    if channels > 1:
        spec_magnitude = spec_magnitude.mean(dim=1, keepdim=True)

    harmonic_sum = torch.log1p(spec_magnitude.clone())
    for k in range(2, max_harmonic + 1):
        indices = harmonic_indices[k][:freq_bins].to(spec_magnitude.device)
        harmonic_k = torch.index_select(spec_magnitude, 2, indices)
        harmonic_sum.add_(torch.log1p(harmonic_k))

    return torch.expm1(harmonic_sum) / max_harmonic


def compute_harmonic_plane_filterbank(
    spec_magnitude: torch.Tensor,
    filterbank: nn.Module,
) -> torch.Tensor:
    """Compute harmonic plane using the learned harmonic filterbank."""
    spec_magnitude = ensure_4d(spec_magnitude)
    if spec_magnitude.shape[1] > 1:
        spec_magnitude = spec_magnitude.mean(dim=1, keepdim=True)
    harmonic_output: torch.Tensor = filterbank(spec_magnitude)
    return harmonic_output.relu_()


def compute_spatial_plane(spec_left: torch.Tensor, spec_right: torch.Tensor) -> torch.Tensor:
    """Compute spatial plane [B, 2, F, T] containing IPD and stereo energy panning."""
    spec_left = ensure_4d(spec_left)
    spec_right = ensure_4d(spec_right)

    ipd = torch.angle(spec_left) - torch.angle(spec_right)
    left_energy = torch.abs(spec_left).square_()
    right_energy = torch.abs(spec_right).square_()
    sum_energy = left_energy + right_energy + 1e-10

    left_energy.sub_(right_energy)
    energy_panning = left_energy.div_(sum_energy)

    ipd = ipd.unsqueeze(1) if ipd.dim() == 3 else ipd
    energy_panning = energy_panning.unsqueeze(1) if energy_panning.dim() == 3 else energy_panning
    return torch.cat([ipd, energy_panning], dim=1)


def compute_psychoacoustic_plane(
    spec_magnitude: torch.Tensor,
    spread_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute psychoacoustic masking threshold plane [B, 1, F, T] via Bark spreading."""
    spec_magnitude = ensure_4d(spec_magnitude)
    if spec_magnitude.shape[1] > 1:
        spec_magnitude = spec_magnitude.mean(dim=1)
    else:
        spec_magnitude = spec_magnitude.squeeze(1)

    mag_clamped = spec_magnitude.clamp(min=1e-10)
    mag_db = 20.0 * torch.log10(mag_clamped)
    mag_db.clamp_(min=-100.0, max=0.0)
    mag_db.add_(100.0).div_(100.0)

    spread = spread_matrix.to(dtype=mag_db.dtype, device=mag_db.device)
    masked = torch.einsum("bft,fg->bgt", mag_db, spread)
    masked.sub_(0.12).clamp_(min=0.0, max=1.0)
    return masked.unsqueeze(1)
