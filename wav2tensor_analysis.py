"""
Unified analysis script for Wav2Tensor audio representation.
Tests different configurations and outputs results to the specified directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor.core import Wav2TensorCore
from wav2tensor.metrics import log_tensor_data


def analyze_audio_segment(
    audio_file: str,
    output_dir: str = "audio",
    start_time: float = 0.0,
    duration: float = 10.0,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    harmonic_method: str = "hps",
    include_planes: list[str] | None = None,
    visualize: bool = True,
    save_tensor_data: bool = True,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    """Analyze an audio segment with Wav2Tensor and output statistics and visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    waveform, file_sample_rate = torchaudio.load(audio_file)
    if file_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)

    start_sample = int(start_time * sample_rate)
    segment_samples = int(duration * sample_rate)
    end_sample = min(start_sample + segment_samples, waveform.shape[1])
    segment = waveform[:, start_sample:end_sample]
    segment_duration = (end_sample - start_sample) / sample_rate

    segment = segment.unsqueeze(0)

    if output_prefix is None:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        segment_info = f"_{int(start_time)}s_to_{int(start_time + segment_duration)}s"
        method_info = f"_{harmonic_method}"
        planes_info = "_reduced" if include_planes is not None else ""
        resolved_prefix = f"{output_dir}/{base_name}{segment_info}{method_info}{planes_info}"
    else:
        resolved_prefix = f"{output_dir}/{output_prefix}"

    start_process_time = time.time()
    wav2tensor = Wav2TensorCore(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        harmonic_method=harmonic_method,
        include_planes=include_planes,
    )
    tensor, planes = wav2tensor(segment)
    process_time = time.time() - start_process_time

    results: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_file": audio_file,
        "segment": {
            "start": float(start_time),
            "duration": float(segment_duration),
            "shape": list(segment.shape[1:]),
        },
        "settings": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "harmonic_method": harmonic_method,
            "include_planes": include_planes,
        },
        "processing_time": process_time,
        "tensor_shape": list(tensor.shape),
    }

    if save_tensor_data:
        log_tensor_data(planes, resolved_prefix, sample_rate, n_fft, results)

    if visualize:
        active_planes = [
            plane_name for plane_name, plane_data in planes.items() if plane_name == "spectral" or torch.max(plane_data) > 0
        ]
        fig = plot_planes(
            planes,
            title=(
                f"Wav2Tensor Analysis: {os.path.basename(audio_file)} "
                f"({start_time:.1f}s to {start_time + segment_duration:.1f}s, "
                f"method: {harmonic_method}, planes: {active_planes})"
            ),
            sample_rate=sample_rate,
            n_fft=n_fft,
        )
        viz_path = f"{resolved_prefix}_planes.png"
        fig.savefig(viz_path, dpi=300)
        plt.close(fig)

    return results


def plot_planes(
    planes: dict[str, torch.Tensor],
    title: str = "Wav2Tensor Planes Analysis",
    sample_rate: int = 22050,
    n_fft: int = 1024,
) -> plt.Figure:
    """Plot all Wav2Tensor planes with enhanced visualization."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    freq_ticks_full = [20, 100, 500, 1000, 2000, 5000, 10000]
    freq_ticks_low = [20, 50, 100, 200, 300, 500, 750, 1000]

    spec_mag = torch.abs(planes["spectral"]).squeeze().detach().cpu().numpy()
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4
    duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 7)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, duration_sec, 7)]

    harm = planes["harmonic"].squeeze().detach().cpu().numpy()
    spatial = planes["spatial"].squeeze().detach().cpu().numpy()
    psycho = planes["psychoacoustic"].squeeze().detach().cpu().numpy()

    im1 = axs[0, 0].imshow(
        10 * np.log10(spec_mag + 1e-8),
        aspect="auto",
        origin="lower",
        extent=[0, spec_mag.shape[1], 0, sample_rate / 2],
    )
    axs[0, 0].set_title("Spectral Magnitude (dB)")
    axs[0, 0].set_ylabel("Frequency (Hz)")
    axs[0, 0].set_yticks(freq_ticks_full)
    axs[0, 0].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[0, 0].set_xticks(time_ticks)
    axs[0, 0].set_xticklabels(time_labels)
    axs[0, 0].set_xlabel("Time (s)")
    plt.colorbar(im1, ax=axs[0, 0])

    log_harm = np.log1p(harm)
    im2 = axs[0, 1].imshow(
        log_harm,
        aspect="auto",
        origin="lower",
        extent=[0, harm.shape[1], 0, 10000],
    )
    harmonic_method = "Filterbank" if harm.max() < 1000 else "HPS"
    axs[0, 1].set_title(f"Harmonic Structure (Log1p, {harmonic_method})")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_yticks(freq_ticks_full)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[0, 1].set_xticks(time_ticks)
    axs[0, 1].set_xticklabels(time_labels)
    axs[0, 1].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axs[0, 1])

    if spatial.shape[0] >= 1:
        ipd = spatial[0]
        im3 = axs[0, 2].imshow(
            ipd,
            aspect="auto",
            origin="lower",
            extent=[0, ipd.shape[1], 0, sample_rate / 2],
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axs[0, 2].set_title("Spatial - Interaural Phase Difference")
        axs[0, 2].set_ylabel("Frequency (Hz)")
        axs[0, 2].set_yticks(freq_ticks_full)
        axs[0, 2].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[0, 2].set_xticks(time_ticks)
        axs[0, 2].set_xticklabels(time_labels)
        axs[0, 2].set_xlabel("Time (s)")
        plt.colorbar(im3, ax=axs[0, 2])
    else:
        axs[0, 2].text(
            0.5,
            0.5,
            "IPD not available\n(mono audio)",
            ha="center",
            va="center",
            transform=axs[0, 2].transAxes,
        )
        axs[0, 2].set_title("Spatial - Interaural Phase Difference")

    im4 = axs[1, 0].imshow(
        harm,
        aspect="auto",
        origin="lower",
        extent=[0, harm.shape[1], 0, 1000],
        vmax=np.percentile(harm, 99.5),
    )
    axs[1, 0].set_title("Harmonic Structure (Raw, 0-1000 Hz)")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_yticks(freq_ticks_low)
    axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks_low])
    axs[1, 0].set_xticks(time_ticks)
    axs[1, 0].set_xticklabels(time_labels)
    axs[1, 0].set_xlabel("Time (s)")
    plt.colorbar(im4, ax=axs[1, 0])

    if spatial.shape[0] >= 2:
        energy = spatial[1]
        im5 = axs[1, 1].imshow(
            energy,
            aspect="auto",
            origin="lower",
            extent=[0, energy.shape[1], 0, sample_rate / 2],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        axs[1, 1].set_title("Spatial - Energy Panning (L/R)")
        axs[1, 1].set_ylabel("Frequency (Hz)")
        axs[1, 1].set_yticks(freq_ticks_full)
        axs[1, 1].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[1, 1].set_xticks(time_ticks)
        axs[1, 1].set_xticklabels(time_labels)
        axs[1, 1].set_xlabel("Time (s)")
        plt.colorbar(im5, ax=axs[1, 1])
    else:
        axs[1, 1].text(
            0.5,
            0.5,
            "Energy panning not available\n(mono audio)",
            ha="center",
            va="center",
            transform=axs[1, 1].transAxes,
        )
        axs[1, 1].set_title("Spatial - Energy Panning (L/R)")

    im6 = axs[1, 2].imshow(
        psycho,
        aspect="auto",
        origin="lower",
        extent=[0, psycho.shape[1], 0, sample_rate / 2],
        vmin=0,
        vmax=1,
    )
    axs[1, 2].set_title("Psychoacoustic Masking")
    axs[1, 2].set_ylabel("Frequency (Hz)")
    axs[1, 2].set_yticks(freq_ticks_full)
    axs[1, 2].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[1, 2].set_xticks(time_ticks)
    axs[1, 2].set_xticklabels(time_labels)
    axs[1, 2].set_xlabel("Time (s)")
    plt.colorbar(im6, ax=axs[1, 2])

    plt.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze audio with Wav2Tensor")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds (default: 10s)")
    parser.add_argument(
        "--method",
        type=str,
        default="hps",
        choices=["hps", "filterbank"],
        help="Harmonic plane method (default: hps)",
    )
    parser.add_argument(
        "--planes",
        type=str,
        default="all",
        help="Planes to include, comma-separated (default: 'all')",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no-data", action="store_true", help="Skip tensor data logging")
    parser.add_argument("--output-dir", type=str, default="audio", help="Output directory (default: audio)")
    parser.add_argument("--prefix", type=str, default=None, help="Custom output file prefix")
    args = parser.parse_args()

    include_planes = None
    if args.planes != "all":
        include_planes = [p.strip() for p in args.planes.split(",")]

    analyze_audio_segment(
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        start_time=args.start,
        duration=args.duration,
        harmonic_method=args.method,
        include_planes=include_planes,
        visualize=not args.no_viz,
        save_tensor_data=not args.no_data,
        output_prefix=args.prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
