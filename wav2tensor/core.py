"""
Core Wav2Tensor encoder/decoder module.
"""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

from wav2tensor.frequency import (
    VALID_PLANES,
    compute_adaptive_frequency_bins,
    compute_psychoacoustic_spreading,
    interpolate_adaptive_spectrum,
)
from wav2tensor.planes import (
    build_harmonic_filterbank,
    compute_harmonic_plane_filterbank,
    compute_harmonic_plane_hps,
    compute_psychoacoustic_plane,
    compute_spatial_plane,
    ensure_4d,
    precompute_harmonic_indices,
)


class Wav2TensorCore(nn.Module):
    """
    Structured multi-plane audio representation tensor T in C^(F x T' x D) with four planes:
    1. Spectral plane: complex STFT representation (real + imag channels in stacked tensor)
    2. Harmonic plane: Harmonic Product Spectrum (HPS) or learned harmonic filterbanks
    3. Spatial plane: Interaural Phase Difference (IPD) and stereo energy panning
    4. Psychoacoustic plane: Bark-scale perceptual masking threshold
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        harmonic_method: str = "hps",
        include_planes: list[str] | None = None,
        use_adaptive_freq: bool = True,
        target_freq_bins: int = 256,
        use_half_precision: bool = False,
        cache_size: int = 8,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_method = harmonic_method
        self.use_adaptive_freq = use_adaptive_freq
        self.use_half_precision = use_half_precision
        self.cache_size = cache_size

        selected_planes = list(include_planes) if include_planes is not None else list(VALID_PLANES)
        for plane in selected_planes:
            if plane not in VALID_PLANES:
                raise ValueError(f"Invalid plane name: {plane}. Must be one of: {', '.join(repr(p) for p in VALID_PLANES)}")
        if "spectral" not in selected_planes:
            selected_planes.insert(0, "spectral")
        self.include_planes = selected_planes

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,
            window_fn=torch.hann_window,
        )

        self.K = 3

        if self.use_adaptive_freq:
            self._setup_adaptive_frequency_mapping(target_freq_bins or 256)

        self._precompute_harmonic_indices()

        if harmonic_method == "filterbank" and "harmonic" in self.include_planes:
            self.harmonic_filterbank = build_harmonic_filterbank(self.K)

        if "psychoacoustic" in self.include_planes:
            self._precompute_psychoacoustic_data()

    def _setup_adaptive_frequency_mapping(self, target_bins: int) -> None:
        adaptive_bins, adaptive_bins_tensor = compute_adaptive_frequency_bins(
            self.sample_rate,
            self.n_fft,
            target_bins,
        )
        self.adaptive_freq_bins = adaptive_bins
        self.target_freq_bins = len(adaptive_bins)
        self.register_buffer("adaptive_freq_bins_tensor", adaptive_bins_tensor)

    def _precompute_harmonic_indices(self) -> None:
        n_freq_bins = self._get_freq_bins()
        harmonic_indices = precompute_harmonic_indices(n_freq_bins, self.K)
        for k, indices in harmonic_indices.items():
            self.register_buffer(f"harmonic_indices_{k}", indices)

    def _get_freq_bins(self) -> int:
        if self.use_adaptive_freq:
            return self.target_freq_bins
        return self.n_fft // 2 + 1

    def _precompute_psychoacoustic_data(self) -> None:
        n_freq_bins = self._get_freq_bins()
        adaptive_bins = self.adaptive_freq_bins if self.use_adaptive_freq else None
        spread, bark_scale = compute_psychoacoustic_spreading(
            self.sample_rate,
            self.n_fft,
            n_freq_bins,
            adaptive_bins,
        )
        self.register_buffer("precomputed_spread", spread)
        self.register_buffer("precomputed_bark_scale", bark_scale)

    def _ensure_4d(self, tensor: torch.Tensor) -> torch.Tensor:
        return ensure_4d(tensor)

    def _compute_spectral_plane_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_samples = waveform.shape
        if channels == 1:
            spec = self.stft(waveform)
        else:
            waveform_flat = waveform.reshape(batch_size * channels, 1, num_samples)
            spec_flat = self.stft(waveform_flat)
            if spec_flat.dim() == 3:
                freq_bins, time_frames = spec_flat.shape[1:]
                spec = spec_flat.reshape(batch_size, channels, freq_bins, time_frames)
            else:
                _, _, freq_bins, time_frames = spec_flat.shape
                spec = spec_flat.reshape(batch_size, channels, 1, freq_bins, time_frames).squeeze(2)

        if self.use_adaptive_freq:
            bins = self.adaptive_freq_bins_tensor.to(spec.device)
            dim = 1 if spec.dim() == 3 else 2
            spec = torch.index_select(spec, dim, bins)

        return spec

    def _compute_harmonic_plane_hps_optimized(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        indices_map = {k: getattr(self, f"harmonic_indices_{k}") for k in range(2, self.K + 1)}
        return compute_harmonic_plane_hps(spec_magnitude, self.K, indices_map)

    def _compute_harmonic_plane_filterbank(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        return compute_harmonic_plane_filterbank(spec_magnitude, self.harmonic_filterbank)

    def _compute_harmonic_plane(self, spec_magnitude: torch.Tensor) -> torch.Tensor:
        if self.harmonic_method == "filterbank":
            return self._compute_harmonic_plane_filterbank(spec_magnitude)
        return self._compute_harmonic_plane_hps_optimized(spec_magnitude)

    def _compute_spatial_plane_optimized(
        self,
        spec_left: torch.Tensor,
        spec_right: torch.Tensor,
    ) -> torch.Tensor:
        return compute_spatial_plane(spec_left, spec_right)

    def _compute_psychoacoustic_plane_optimized(
        self,
        spec_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        return compute_psychoacoustic_plane(spec_magnitude, self.precomputed_spread)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Transform input waveform (B, C, T) into structured multi-plane tensor and plane dictionary."""
        original_dtype = waveform.dtype
        if self.use_half_precision and waveform.dtype != torch.float16:
            waveform = waveform.half()

        _, channels, _ = waveform.shape
        planes: dict[str, torch.Tensor] = {}
        tensor_components: list[torch.Tensor] = []

        if channels == 1 or "spatial" not in self.include_planes:
            spec = self._compute_spectral_plane_batch(waveform)
            if spec.dim() == 4 and spec.shape[1] > 1:
                spec = spec.mean(dim=1, keepdim=True)
            spec_left = spec_right = spec
        else:
            spec_left = self._compute_spectral_plane_batch(waveform[:, 0:1])
            spec_right = self._compute_spectral_plane_batch(waveform[:, 1:2])
            spec = (spec_left + spec_right) / 2.0

        planes["spectral"] = spec

        spec_real = ensure_4d(torch.real(spec))
        spec_imag = ensure_4d(torch.imag(spec))
        tensor_components.extend([spec_real, spec_imag])

        spec_magnitude = torch.abs(spec)

        if "harmonic" in self.include_planes:
            harmonic_plane = self._compute_harmonic_plane(spec_magnitude)
            planes["harmonic"] = harmonic_plane
            tensor_components.append(harmonic_plane)

        if "spatial" in self.include_planes:
            spatial_plane = self._compute_spatial_plane_optimized(spec_left, spec_right)
            planes["spatial"] = spatial_plane
            tensor_components.append(spatial_plane)

        if "psychoacoustic" in self.include_planes:
            psychoacoustic_plane = self._compute_psychoacoustic_plane_optimized(spec_magnitude)
            planes["psychoacoustic"] = psychoacoustic_plane
            tensor_components.append(psychoacoustic_plane)

        tensor = torch.cat(tensor_components, dim=1)

        if self.use_half_precision and original_dtype != torch.float16:
            tensor = tensor.to(dtype=original_dtype)
            for key, value in planes.items():
                if not torch.is_complex(value):
                    planes[key] = value.to(dtype=original_dtype)

        return tensor, planes

    def inverse(
        self,
        tensor: torch.Tensor,
        planes: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Convert Wav2Tensor representation back to time-domain waveform."""
        tensor = ensure_4d(tensor)

        if planes is None:
            planes = {}
            current_channel = 0
            spec_real = tensor[:, current_channel : current_channel + 1]
            spec_imag = tensor[:, current_channel + 1 : current_channel + 2]
            spec_real = spec_real.squeeze(1) if spec_real.shape[1] == 1 else spec_real
            spec_imag = spec_imag.squeeze(1) if spec_imag.shape[1] == 1 else spec_imag
            planes["spectral"] = torch.complex(spec_real, spec_imag)
            current_channel += 2

            if "harmonic" in self.include_planes and current_channel < tensor.shape[1]:
                planes["harmonic"] = tensor[:, current_channel : current_channel + 1]
                current_channel += 1

            if "spatial" in self.include_planes and current_channel + 1 < tensor.shape[1]:
                planes["spatial"] = tensor[:, current_channel : current_channel + 2]
                current_channel += 2

            if "psychoacoustic" in self.include_planes and current_channel < tensor.shape[1]:
                planes["psychoacoustic"] = tensor[:, current_channel : current_channel + 1]
                current_channel += 1

        spec = planes["spectral"]

        if self.use_adaptive_freq:
            spec = interpolate_adaptive_spectrum(
                spec,
                self.n_fft,
                self.adaptive_freq_bins,
                self.adaptive_freq_bins_tensor,
            )

        if spec.dim() == 4:
            spec = spec.squeeze(1)

        window = torch.hann_window(self.n_fft, device=spec.device)
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=False,
        )

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if "spatial" in planes and "harmonic" in planes and not self.use_adaptive_freq:
            spatial = planes["spatial"]
            ipd = spatial[:, 0]
            energy_panning = spatial[:, 1]
            harmonic = planes["harmonic"].squeeze(1) if planes["harmonic"].dim() == 4 else planes["harmonic"]

            spec_enhanced = spec * (1.0 + harmonic)
            spec_left = spec_enhanced * torch.exp(1j * ipd / 2.0)
            spec_right = spec_enhanced * torch.exp(-1j * ipd / 2.0)

            energy_left = torch.clamp((1.0 + energy_panning) / 2.0, min=0.0)
            energy_right = torch.clamp((1.0 - energy_panning) / 2.0, min=0.0)

            spec_left = spec_left * torch.sqrt(energy_left)
            spec_right = spec_right * torch.sqrt(energy_right)

            left_wave = torch.istft(
                spec_left,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=False,
            )
            right_wave = torch.istft(
                spec_right,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=False,
            )
            waveform = torch.stack([left_wave, right_wave], dim=1)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        return waveform
