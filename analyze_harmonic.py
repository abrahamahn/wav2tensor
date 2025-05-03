"""
Analyze harmonic structure with enhanced visualization using log compression.
This helps reveal the harmonic content that may be obscured by high dynamic range.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor import Wav2TensorCore

def log_tensor_data(planes, output_prefix, sample_rate=22050, n_fft=1024):
    """Log meaningful tensor data to a file for further analysis."""
    # Create a log file
    log_path = f"{output_prefix}_tensor_data.log"
    
    # Get harmonic plane
    harm = planes['harmonic'].squeeze().cpu().numpy()
    spec_mag = torch.abs(planes['spectral']).squeeze().cpu().numpy()
    
    # Calculate frequency bins in Hz
    freqs = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
    # Calculate time frames in seconds
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4  # Assuming default hop_length = n_fft/4
    duration_sec = time_frames * hop_length / sample_rate
    times = np.linspace(0, duration_sec, time_frames)
    
    # Prepare data for logging
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "harmonic_plane": {
            "shape": harm.shape,
            "min": float(np.min(harm)),
            "max": float(np.max(harm)),
            "mean": float(np.mean(harm)),
            "median": float(np.median(harm)),
            "std": float(np.std(harm)),
            "percentiles": {
                "1%": float(np.percentile(harm, 1)),
                "5%": float(np.percentile(harm, 5)),
                "25%": float(np.percentile(harm, 25)),
                "50%": float(np.percentile(harm, 50)),
                "75%": float(np.percentile(harm, 75)),
                "95%": float(np.percentile(harm, 95)),
                "99%": float(np.percentile(harm, 99)),
            }
        },
        "spectral_plane": {
            "shape": spec_mag.shape,
            "min": float(np.min(spec_mag)),
            "max": float(np.max(spec_mag)),
            "mean": float(np.mean(spec_mag)),
            "median": float(np.median(spec_mag)),
        },
    }
    
    # Find peaks in harmonic plane
    # Get position of the maximum value
    max_idx = np.unravel_index(np.argmax(harm), harm.shape)
    max_freq = freqs[max_idx[0]]
    max_time = times[max_idx[1]] if max_idx[1] < len(times) else times[-1]
    
    # Find harmonic activity in specific frequency bands
    bands = [
        (20, 100),    # Sub-bass
        (100, 250),   # Bass
        (250, 500),   # Low mids
        (500, 2000),  # Mids
        (2000, 5000), # Upper mids
        (5000, 10000) # Highs
    ]
    
    band_data = {}
    for band_name, (low_freq, high_freq) in zip(
        ["sub_bass", "bass", "low_mids", "mids", "upper_mids", "highs"], bands
    ):
        # Find indices corresponding to this frequency band
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
                "time": float(times[np.argmax(np.max(band_values, axis=0))])
            }
        }
    
    data["harmonic_peaks"] = {
        "global_max": {
            "value": float(harm[max_idx]),
            "frequency": float(max_freq),
            "time": float(max_time)
        },
        "frequency_bands": band_data
    }
    
    # Sample harmonic values at specific points
    sample_points = []
    # Sample at lower frequencies (0-1000 Hz)
    for freq in [50, 100, 250, 500, 750, 1000]:
        freq_idx = np.argmin(np.abs(freqs - freq))
        # Sample at beginning, middle and end of time
        for t_idx in [0, time_frames // 2, time_frames - 1]:
            if t_idx < harm.shape[1]:
                sample_points.append({
                    "frequency": float(freqs[freq_idx]),
                    "time": float(times[t_idx] if t_idx < len(times) else times[-1]),
                    "value": float(harm[freq_idx, t_idx])
                })
    
    data["sample_points"] = sample_points
    
    # Save as formatted text file for easier reading
    with open(log_path, 'w') as f:
        f.write("# Wav2Tensor Harmonic Plane Data\n")
        f.write(f"# Generated: {data['timestamp']}\n\n")
        
        f.write("## General Statistics\n")
        f.write(f"Harmonic Plane Shape: {data['harmonic_plane']['shape']}\n")
        f.write(f"Min: {data['harmonic_plane']['min']:.6f}\n")
        f.write(f"Max: {data['harmonic_plane']['max']:.6f}\n")
        f.write(f"Mean: {data['harmonic_plane']['mean']:.6f}\n")
        f.write(f"Median: {data['harmonic_plane']['median']:.6f}\n")
        f.write(f"Standard Deviation: {data['harmonic_plane']['std']:.6f}\n\n")
        
        f.write("## Percentiles\n")
        for pct, value in data['harmonic_plane']['percentiles'].items():
            f.write(f"{pct}: {value:.6f}\n")
        f.write("\n")
        
        f.write("## Harmonic Peaks\n")
        f.write(f"Global Maximum: {data['harmonic_peaks']['global_max']['value']:.6f} at " +
                f"{data['harmonic_peaks']['global_max']['frequency']:.2f} Hz, " +
                f"{data['harmonic_peaks']['global_max']['time']:.2f} s\n\n")
        
        f.write("## Frequency Band Analysis\n")
        for band_name, band_info in data['harmonic_peaks']['frequency_bands'].items():
            f.write(f"### {band_name.replace('_', ' ').title()} ({band_info['freq_range']})\n")
            f.write(f"Mean: {band_info['mean']:.6f}\n")
            f.write(f"Max: {band_info['max']:.6f}\n")
            f.write(f"Max Location: {band_info['max_location']['freq']:.2f} Hz at {band_info['max_location']['time']:.2f} s\n\n")
        
        f.write("## Sample Points (Frequency, Time, Value)\n")
        for point in data['sample_points']:
            f.write(f"{point['frequency']:.2f} Hz, {point['time']:.2f} s: {point['value']:.6f}\n")
    
    # Also save as JSON for programmatic access
    json_path = f"{output_prefix}_tensor_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Tensor data logged to {log_path}")
    print(f"JSON data saved to {json_path}")

def plot_harmonic_plane(planes, title="Harmonic Structure Analysis", sample_rate=22050, n_fft=1024):
    """Plot the harmonic plane with different visualization methods."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    # Calculate frequency axis in Hz
    freqs = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
    # Add meaningful frequency ticks for full range
    freq_ticks_full = [20, 100, 500, 1000, 2000, 5000, 10000]
    
    # Add meaningful frequency ticks for low range (0-1000 Hz)
    freq_ticks_low = [20, 50, 100, 200, 300, 500, 750, 1000]
    
    # Add time ticks in seconds
    spec_mag = torch.abs(planes['spectral']).squeeze().cpu().numpy()
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4  # Assuming default hop_length = n_fft/4
    duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 7)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, duration_sec, 7)]
    
    # Get harmonic plane
    harm = planes['harmonic'].squeeze().cpu().numpy()
    
    # 1. Spectral magnitude (reference)
    im1 = axs[0, 0].imshow(
        10 * np.log10(spec_mag + 1e-8), 
        aspect='auto', 
        origin='lower',
        extent=[0, spec_mag.shape[1], 0, sample_rate/2]
    )
    axs[0, 0].set_title("Spectral Magnitude (dB)")
    axs[0, 0].set_ylabel("Frequency (Hz)")
    axs[0, 0].set_yticks(freq_ticks_full)
    axs[0, 0].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[0, 0].set_xticks(time_ticks)
    axs[0, 0].set_xticklabels(time_labels)
    axs[0, 0].set_xlabel("Time (s)")
    plt.colorbar(im1, ax=axs[0, 0])
    
    # 2. Raw harmonic plane (focused on 0-1000 Hz)
    im2 = axs[0, 1].imshow(
        harm, 
        aspect='auto', 
        origin='lower',
        extent=[0, harm.shape[1], 0, 1000],  # Limited to 0-1000 Hz
        vmax=np.percentile(harm, 99.5)  # Clip extreme values for better visualization
    )
    axs[0, 1].set_title(f"Harmonic Structure (Raw, K={planes['harmonic'].shape[1]}) 0-1000 Hz")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_yticks(freq_ticks_low)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks_low])
    axs[0, 1].set_xticks(time_ticks)
    axs[0, 1].set_xticklabels(time_labels)
    axs[0, 1].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axs[0, 1])
    
    # 3. Log-compressed harmonic plane (log1p)
    log_harm = np.log1p(harm)
    im3 = axs[1, 0].imshow(
        log_harm, 
        aspect='auto', 
        origin='lower',
        extent=[0, harm.shape[1], 0, 10000]  # Show up to 10000 Hz
    )
    axs[1, 0].set_title("Harmonic Structure (Log1p) 0-10000 Hz")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_yticks(freq_ticks_full)
    axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[1, 0].set_xticks(time_ticks)
    axs[1, 0].set_xticklabels(time_labels)
    axs[1, 0].set_xlabel("Time (s)")
    plt.colorbar(im3, ax=axs[1, 0])
    
    # 4. Normalized harmonic plane (relative strength) - focused on 0-1000 Hz
    norm_harm = harm / np.maximum(np.max(harm, axis=0, keepdims=True), 1e-10)
    im4 = axs[1, 1].imshow(
        norm_harm, 
        aspect='auto', 
        origin='lower',
        extent=[0, harm.shape[1], 0, 1000],  # Limited to 0-1000 Hz
        vmin=0, 
        vmax=1
    )
    axs[1, 1].set_title("Harmonic Structure (Normalized) 0-1000 Hz")
    axs[1, 1].set_ylabel("Frequency (Hz)")
    axs[1, 1].set_yticks(freq_ticks_low)
    axs[1, 1].set_yticklabels([f"{f}" for f in freq_ticks_low])
    axs[1, 1].set_xticks(time_ticks)
    axs[1, 1].set_xticklabels(time_labels)
    axs[1, 1].set_xlabel("Time (s)")
    plt.colorbar(im4, ax=axs[1, 1])
    
    # Print statistics
    print(f"Harmonic plane statistics:")
    print(f"  Shape: {harm.shape}")
    print(f"  Min: {np.min(harm):.6f}")
    print(f"  Max: {np.max(harm):.6f}")
    print(f"  Mean: {np.mean(harm):.6f}")
    print(f"  99th percentile: {np.percentile(harm, 99):.6f}")
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze harmonic structure with enhanced visualization")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds to analyze (default: 5s)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    args = parser.parse_args()
    
    # Load the audio file
    try:
        waveform, sample_rate = torchaudio.load(args.audio_file)
        print(f"Loaded audio: {args.audio_file}")
        
        # Calculate samples for the segment
        start_sample = int(args.start * sample_rate)
        segment_samples = int(args.duration * sample_rate)
        
        # Extract the segment
        end_sample = min(start_sample + segment_samples, waveform.shape[1])
        segment = waveform[:, start_sample:end_sample]
        segment_duration = (end_sample - start_sample) / sample_rate
        
        print(f"Analyzing segment from {args.start:.1f}s to {args.start + segment_duration:.1f}s ({segment_duration:.1f}s duration)")
        print(f"Segment shape: {segment.shape}")
        
        # Add batch dimension if needed
        segment = segment.unsqueeze(0)  # [B, C, T]
        
        # Initialize Wav2Tensor encoder
        wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
        
        # Convert to Wav2Tensor representation
        tensor, planes = wav2tensor(segment)
        
        # Generate output path
        base_name = os.path.splitext(args.audio_file)[0]
        segment_info = f"_{int(args.start)}s_to_{int(args.start + segment_duration)}s"
        output_prefix = f"{base_name}{segment_info}"
        output_path = f"{output_prefix}_harmonic.png"
        
        # Log tensor data
        log_tensor_data(
            planes,
            output_prefix,
            sample_rate=sample_rate,
            n_fft=wav2tensor.n_fft
        )
        
        # Plot the harmonic plane with enhanced visualization
        title = f"Harmonic Analysis: {os.path.basename(args.audio_file)} ({args.start:.1f}s to {args.start + segment_duration:.1f}s)"
        fig = plot_harmonic_plane(
            planes, 
            title=title,
            sample_rate=sample_rate,
            n_fft=wav2tensor.n_fft
        )
        
        # Save the plot
        fig.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return

if __name__ == "__main__":
    main() 