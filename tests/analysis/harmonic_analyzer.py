"""
Analyze harmonic structure with enhanced visualization using log compression.
This helps reveal the harmonic content that may be obscured by high dynamic range.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wav2tensor import Wav2TensorCore
from wav2tensor.metrics import log_tensor_data


def plot_harmonic_plane(
    planes: dict[str, torch.Tensor],
    title: str = "Harmonic Structure Analysis",
    sample_rate: int = 22050,
    n_fft: int = 1024,
) -> plt.Figure:
    """Plot the harmonic plane with different visualization methods."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    freq_ticks_full = [20, 100, 500, 1000, 2000, 5000, 10000]
    freq_ticks_low = [20, 50, 100, 200, 300, 500, 750, 1000]

    spec_mag = torch.abs(planes["spectral"]).squeeze().cpu().numpy()
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4
    duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 7)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, duration_sec, 7)]

    harm = planes["harmonic"].squeeze().cpu().numpy()

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

    im2 = axs[0, 1].imshow(
        harm,
        aspect="auto",
        origin="lower",
        extent=[0, harm.shape[1], 0, 1000],
        vmax=np.percentile(harm, 99.5),
    )
    axs[0, 1].set_title(f"Harmonic Structure (Raw, K={planes['harmonic'].shape[1]}) 0-1000 Hz")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_yticks(freq_ticks_low)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks_low])
    axs[0, 1].set_xticks(time_ticks)
    axs[0, 1].set_xticklabels(time_labels)
    axs[0, 1].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axs[0, 1])

    log_harm = np.log1p(harm)
    im3 = axs[1, 0].imshow(
        log_harm,
        aspect="auto",
        origin="lower",
        extent=[0, harm.shape[1], 0, 10000],
    )
    axs[1, 0].set_title("Harmonic Structure (Log1p) 0-10000 Hz")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_yticks(freq_ticks_full)
    axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[1, 0].set_xticks(time_ticks)
    axs[1, 0].set_xticklabels(time_labels)
    axs[1, 0].set_xlabel("Time (s)")
    plt.colorbar(im3, ax=axs[1, 0])

    norm_harm = harm / np.maximum(np.max(harm, axis=0, keepdims=True), 1e-10)
    im4 = axs[1, 1].imshow(
        norm_harm,
        aspect="auto",
        origin="lower",
        extent=[0, harm.shape[1], 0, 1000],
        vmin=0,
        vmax=1,
    )
    axs[1, 1].set_title("Harmonic Structure (Normalized) 0-1000 Hz")
    axs[1, 1].set_ylabel("Frequency (Hz)")
    axs[1, 1].set_yticks(freq_ticks_low)
    axs[1, 1].set_yticklabels([f"{f}" for f in freq_ticks_low])
    axs[1, 1].set_xticks(time_ticks)
    axs[1, 1].set_xticklabels(time_labels)
    axs[1, 1].set_xlabel("Time (s)")
    plt.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze harmonic structure with enhanced visualization")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds to analyze (default: 5s)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    parser.add_argument("--output-dir", type=str, help="Output directory for results", default="results")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive frequency resolution")
    parser.add_argument(
        "--target-freq-bins",
        type=int,
        default=256,
        help="Number of frequency bins with adaptive mode",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    waveform, sample_rate = torchaudio.load(args.audio_file)

    start_sample = int(args.start * sample_rate)
    segment_samples = int(args.duration * sample_rate)
    end_sample = min(start_sample + segment_samples, waveform.shape[1])
    segment = waveform[:, start_sample:end_sample]
    segment_duration = (end_sample - start_sample) / sample_rate
    segment = segment.unsqueeze(0)

    wav2tensor = Wav2TensorCore(
        sample_rate=sample_rate,
        use_adaptive_freq=args.adaptive,
        target_freq_bins=args.target_freq_bins,
    )
    _, planes = wav2tensor(segment)

    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    segment_info = f"_{int(args.start)}s_to_{int(args.start + segment_duration)}s"
    adaptive_info = "_adaptive" if args.adaptive else ""
    output_prefix = os.path.join(args.output_dir, f"{base_name}{segment_info}{adaptive_info}")
    output_path = f"{output_prefix}_harmonic.png"

    log_tensor_data(
        planes,
        output_prefix,
        sample_rate=sample_rate,
        n_fft=wav2tensor.n_fft,
    )

    title = (
        f"Harmonic Analysis: {os.path.basename(args.audio_file)} ({args.start:.1f}s to {args.start + segment_duration:.1f}s)"
    )
    if args.adaptive:
        title += f" [Adaptive: {planes['harmonic'].shape[2]} bins]"

    fig = plot_harmonic_plane(
        planes,
        title=title,
        sample_rate=sample_rate,
        n_fft=wav2tensor.n_fft,
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
