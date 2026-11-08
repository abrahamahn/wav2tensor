"""
Audio reconstruction metrics and multi-plane tensor statistical analysis utilities.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import torch

FREQUENCY_BANDS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("sub_bass", (20, 100)),
    ("bass", (100, 250)),
    ("low_mids", (250, 500)),
    ("mids", (500, 2000)),
    ("upper_mids", (2000, 5000)),
    ("highs", (5000, 10000)),
)


def calculate_mse(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Calculate Mean Squared Error between clean and reconstructed signals."""
    return float(np.mean((clean - enhanced) ** 2))


def calculate_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in decibels."""
    noise = clean - enhanced
    signal_power = float(np.sum(clean**2))
    noise_power = float(np.sum(noise**2))
    if noise_power > 0.0:
        return float(10.0 * np.log10(signal_power / noise_power))
    return 100.0


def calculate_lsd(clean_spec: np.ndarray, enhanced_spec: np.ndarray) -> float:
    """Calculate Log-Spectral Distance (LSD) between magnitude spectrograms."""
    eps = 1e-8
    clean_log = np.log10(np.abs(clean_spec) + eps)
    enhanced_log = np.log10(np.abs(enhanced_spec) + eps)
    return float(np.mean(np.sqrt(np.mean((clean_log - enhanced_log) ** 2, axis=0))))


def summarize_tensor_planes(
    planes: dict[str, torch.Tensor],
    sample_rate: int = 22050,
    n_fft: int = 1024,
    results_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute summary statistics, percentiles, and band peaks across Wav2Tensor planes."""
    spec_mag = torch.abs(planes["spectral"]).squeeze().detach().cpu().numpy()
    harm = planes["harmonic"].squeeze().detach().cpu().numpy()
    spatial_tensor = planes.get("spatial")
    spatial = (
        spatial_tensor.squeeze().detach().cpu().numpy() if spatial_tensor is not None and spatial_tensor.shape[1] > 0 else None
    )
    psycho_tensor = planes.get("psychoacoustic")
    psycho = psycho_tensor.squeeze().detach().cpu().numpy() if psycho_tensor is not None else None

    freqs = np.linspace(0, sample_rate / 2, harm.shape[0])
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4
    duration_sec = time_frames * hop_length / sample_rate
    times = np.linspace(0, duration_sec, time_frames)

    data: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tensor_shapes": {
            "spectral": list(spec_mag.shape),
            "harmonic": list(harm.shape),
            "spatial": None if spatial is None else list(spatial.shape),
            "psychoacoustic": None if psycho is None else list(psycho.shape),
        },
        "harmonic_plane": {
            "shape": list(harm.shape),
            "min": float(np.min(harm)),
            "max": float(np.max(harm)),
            "mean": float(np.mean(harm)),
            "median": float(np.median(harm)),
            "std": float(np.std(harm)),
            "percentiles": {f"{pct}%": float(np.percentile(harm, pct)) for pct in (1, 5, 25, 50, 75, 95, 99)},
        },
        "spectral_plane": {
            "shape": list(spec_mag.shape),
            "min": float(np.min(spec_mag)),
            "max": float(np.max(spec_mag)),
            "mean": float(np.mean(spec_mag)),
            "median": float(np.median(spec_mag)),
        },
    }

    if psycho is not None:
        data["psychoacoustic_plane"] = {
            "shape": list(psycho.shape),
            "min": float(np.min(psycho)),
            "max": float(np.max(psycho)),
            "mean": float(np.mean(psycho)),
        }

    if spatial is not None:
        data["spatial_plane"] = {
            "shape": list(spatial.shape),
            "min": float(np.min(spatial)),
            "max": float(np.max(spatial)),
            "mean": float(np.mean(spatial)),
            "ipd_mean": float(np.mean(spatial[0])) if spatial.shape[0] > 0 else None,
            "energy_mean": float(np.mean(spatial[1])) if spatial.shape[0] > 1 else None,
        }

    max_idx = np.unravel_index(np.argmax(harm), harm.shape)
    max_freq = freqs[max_idx[0]]
    max_time = times[max_idx[1]] if max_idx[1] < len(times) else times[-1]

    band_data: dict[str, Any] = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS:
        band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
        if len(band_indices) == 0:
            continue
        band_values = harm[band_indices, :]
        band_data[band_name] = {
            "freq_range": f"{low_freq}-{high_freq} Hz",
            "mean": float(np.mean(band_values)),
            "max": float(np.max(band_values)),
            "max_location": {
                "freq": float(freqs[band_indices[np.argmax(np.max(band_values, axis=1))]]),
                "time": float(times[np.argmax(np.max(band_values, axis=0))]),
            },
        }

    data["harmonic_peaks"] = {
        "global_max": {
            "value": float(harm[max_idx]),
            "frequency": float(max_freq),
            "time": float(max_time),
        },
        "frequency_bands": band_data,
    }

    sample_points = []
    for freq in (50, 100, 250, 500, 750, 1000):
        freq_idx = int(np.argmin(np.abs(freqs - freq)))
        for t_idx in (0, time_frames // 2, time_frames - 1):
            if t_idx < harm.shape[1]:
                sample_points.append(
                    {
                        "frequency": float(freqs[freq_idx]),
                        "time": float(times[t_idx] if t_idx < len(times) else times[-1]),
                        "value": float(harm[freq_idx, t_idx]),
                    }
                )
    data["sample_points"] = sample_points

    if results_dict is not None:
        data["analysis_metadata"] = results_dict

    return data


def log_tensor_data(
    planes: dict[str, torch.Tensor],
    output_prefix: str,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    results_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write formatted markdown/log and JSON summaries of Wav2Tensor plane statistics."""
    data = summarize_tensor_planes(planes, sample_rate=sample_rate, n_fft=n_fft, results_dict=results_dict)
    log_path = f"{output_prefix}_tensor_data.log"
    json_path = f"{output_prefix}_tensor_data.json"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Wav2Tensor Data Analysis\n")
        f.write(f"# Generated: {data['timestamp']}\n\n")

        if results_dict:
            f.write("## Configuration\n")
            f.write(f"Audio file: {results_dict.get('audio_file', 'unknown')}\n")
            seg = results_dict.get("segment", {})
            start = seg.get("start", 0.0)
            dur = seg.get("duration", 0.0)
            f.write(f"Segment: {start:.1f}s to {start + dur:.1f}s\n")
            f.write(f"Harmonic method: {results_dict.get('settings', {}).get('harmonic_method', 'unknown')}\n")
            f.write(f"Processing time: {results_dict.get('processing_time', 0.0):.2f}s\n\n")

        f.write("## Tensor Shapes\n")
        f.writelines(f"{name.title()}: {shape}\n" for name, shape in data["tensor_shapes"].items())
        f.write("\n")

        f.write("## Harmonic Plane Statistics\n")
        f.write(f"Min: {data['harmonic_plane']['min']:.6f}\n")
        f.write(f"Max: {data['harmonic_plane']['max']:.6f}\n")
        f.write(f"Mean: {data['harmonic_plane']['mean']:.6f}\n")
        f.write(f"Median: {data['harmonic_plane']['median']:.6f}\n")
        f.write(f"Standard Deviation: {data['harmonic_plane']['std']:.6f}\n\n")

        f.write("## Percentiles\n")
        f.writelines(f"{pct}: {value:.6f}\n" for pct, value in data["harmonic_plane"]["percentiles"].items())
        f.write("\n")

        global_max = data["harmonic_peaks"]["global_max"]
        f.write("## Harmonic Peaks\n")
        f.write(
            f"Global Maximum: {global_max['value']:.6f} at {global_max['frequency']:.2f} Hz, {global_max['time']:.2f} s\n\n"
        )

        f.write("## Frequency Band Analysis\n")
        for band_name, band_info in data["harmonic_peaks"]["frequency_bands"].items():
            f.write(f"### {band_name.replace('_', ' ').title()} ({band_info['freq_range']})\n")
            f.write(f"Mean: {band_info['mean']:.6f}\n")
            f.write(f"Max: {band_info['max']:.6f}\n")
            f.write(
                f"Max Location: {band_info['max_location']['freq']:.2f} Hz at {band_info['max_location']['time']:.2f} s\n\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data
