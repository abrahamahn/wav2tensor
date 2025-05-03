"""
Unified analysis script for Wav2Tensor audio representation.
Tests different configurations and outputs results to the audio folder.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime
import time

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor.core import Wav2TensorCore

def analyze_audio_segment(
    audio_file, 
    output_dir='audio',
    start_time=0,
    duration=10,
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    harmonic_method='hps',
    include_planes=None,
    visualize=True,
    save_tensor_data=True,
    output_prefix=None
):
    """
    Analyze an audio segment with Wav2Tensor and output results.
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save outputs
        start_time: Start time in seconds
        duration: Duration in seconds
        sample_rate: Audio sample rate
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        harmonic_method: 'hps' or 'filterbank'
        include_planes: List of planes to include, or None for all planes
        visualize: Whether to generate visualizations
        save_tensor_data: Whether to save tensor data
        output_prefix: Custom prefix for output files
    
    Returns:
        Dictionary with analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio file
    print(f"Loading audio: {audio_file}")
    waveform, file_sample_rate = torchaudio.load(audio_file)
    
    # Resample if needed
    if file_sample_rate != sample_rate:
        print(f"Resampling from {file_sample_rate}Hz to {sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)
    
    # Calculate samples for the segment
    start_sample = int(start_time * sample_rate)
    segment_samples = int(duration * sample_rate)
    
    # Extract the segment
    end_sample = min(start_sample + segment_samples, waveform.shape[1])
    segment = waveform[:, start_sample:end_sample]
    segment_duration = (end_sample - start_sample) / sample_rate
    
    print(f"Analyzing segment from {start_time:.1f}s to {start_time + segment_duration:.1f}s ({segment_duration:.1f}s duration)")
    print(f"Segment shape: {segment.shape}")
    
    # Add batch dimension if needed
    segment = segment.unsqueeze(0)  # [B, C, T]
    
    # Create filename prefix
    if output_prefix is None:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        segment_info = f"_{int(start_time)}s_to_{int(start_time + segment_duration)}s"
        method_info = f"_{harmonic_method}"
        planes_info = "_reduced" if include_planes is not None else ""
        output_prefix = f"{output_dir}/{base_name}{segment_info}{method_info}{planes_info}"
    else:
        output_prefix = f"{output_dir}/{output_prefix}"
    
    # Initialize timer
    start_process_time = time.time()
    
    # Initialize Wav2Tensor encoder
    plane_str = "all planes" if include_planes is None else f"planes: {include_planes}"
    print(f"Initializing Wav2TensorCore with harmonic_method='{harmonic_method}', {plane_str}")
    wav2tensor = Wav2TensorCore(
        sample_rate=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        harmonic_method=harmonic_method,
        include_planes=include_planes
    )
    
    # Convert to Wav2Tensor representation
    tensor, planes = wav2tensor(segment)
    
    # Calculate processing time
    process_time = time.time() - start_process_time
    print(f"Processed in {process_time:.2f} seconds")
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_file": audio_file,
        "segment": {
            "start": float(start_time),
            "duration": float(segment_duration),
            "shape": segment.shape[1:],  # Exclude batch dimension
        },
        "settings": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "harmonic_method": harmonic_method,
            "include_planes": include_planes
        },
        "processing_time": process_time,
        "tensor_shape": tensor.shape
    }
    
    # Save tensor data
    if save_tensor_data:
        log_tensor_data(planes, output_prefix, sample_rate, n_fft, results)
    
    # Generate visualizations
    if visualize:
        # Determine which planes are actually available with data (not just placeholders)
        active_planes = []
        for plane_name, plane_data in planes.items():
            if plane_name == 'spectral' or torch.max(plane_data) > 0:
                active_planes.append(plane_name)
        
        fig = plot_planes(
            planes, 
            title=f"Wav2Tensor Analysis: {os.path.basename(audio_file)} "
                  f"({start_time:.1f}s to {start_time + segment_duration:.1f}s, "
                  f"method: {harmonic_method}, planes: {active_planes})",
            sample_rate=sample_rate,
            n_fft=n_fft
        )
        
        # Save the plot
        viz_path = f"{output_prefix}_planes.png"
        fig.savefig(viz_path, dpi=300)
        print(f"Visualization saved to {viz_path}")
        
        # Close figure to free memory
        plt.close(fig)
    
    return results

def log_tensor_data(planes, output_prefix, sample_rate, n_fft, results_dict=None):
    """Log meaningful tensor data to a file for further analysis."""
    # Create a log file
    log_path = f"{output_prefix}_tensor_data.log"
    
    # Get planes and detach tensors to handle parameters that require grad
    spec_mag = torch.abs(planes['spectral']).squeeze().detach().cpu().numpy()
    harm = planes['harmonic'].squeeze().detach().cpu().numpy()
    spatial = planes['spatial'].squeeze().detach().cpu().numpy() if planes['spatial'].shape[1] > 0 else None
    psycho = planes['psychoacoustic'].squeeze().detach().cpu().numpy()
    
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
        "tensor_shapes": {
            "spectral": spec_mag.shape,
            "harmonic": harm.shape,
            "spatial": None if spatial is None else spatial.shape,
            "psychoacoustic": psycho.shape
        },
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
        "psychoacoustic_plane": {
            "shape": psycho.shape,
            "min": float(np.min(psycho)),
            "max": float(np.max(psycho)),
            "mean": float(np.mean(psycho)),
        }
    }
    
    # Add spatial plane stats if available
    if spatial is not None:
        data["spatial_plane"] = {
            "shape": spatial.shape,
            "min": float(np.min(spatial)),
            "max": float(np.max(spatial)),
            "mean": float(np.mean(spatial)),
            "ipd_mean": float(np.mean(spatial[0])) if spatial.shape[0] > 0 else None,
            "energy_mean": float(np.mean(spatial[1])) if spatial.shape[0] > 1 else None,
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
    
    # Add results from analysis if provided
    if results_dict:
        data["analysis_metadata"] = results_dict
    
    # Save as formatted text file for easier reading
    with open(log_path, 'w') as f:
        f.write("# Wav2Tensor Data Analysis\n")
        f.write(f"# Generated: {data['timestamp']}\n\n")
        
        f.write("## Configuration\n")
        if results_dict:
            f.write(f"Audio file: {results_dict.get('audio_file', 'unknown')}\n")
            f.write(f"Segment: {results_dict.get('segment', {}).get('start', 0):.1f}s to ")
            f.write(f"{results_dict.get('segment', {}).get('start', 0) + results_dict.get('segment', {}).get('duration', 0):.1f}s\n")
            f.write(f"Harmonic method: {results_dict.get('settings', {}).get('harmonic_method', 'unknown')}\n")
            f.write(f"Processing time: {results_dict.get('processing_time', 0):.2f}s\n\n")
        
        f.write("## Tensor Shapes\n")
        for name, shape in data['tensor_shapes'].items():
            f.write(f"{name.title()}: {shape}\n")
        f.write("\n")
        
        f.write("## Harmonic Plane Statistics\n")
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
    
    # Also save as JSON for programmatic access
    json_path = f"{output_prefix}_tensor_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Tensor data logged to {log_path}")
    print(f"JSON data saved to {json_path}")

def plot_planes(planes, title="Wav2Tensor Planes Analysis", sample_rate=22050, n_fft=1024):
    """Plot all Wav2Tensor planes with enhanced visualization."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)
    
    # Calculate frequency axis in Hz
    freqs = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
    # Add meaningful frequency ticks for different ranges
    freq_ticks_full = [20, 100, 500, 1000, 2000, 5000, 10000]
    freq_ticks_low = [20, 50, 100, 200, 300, 500, 750, 1000]
    
    # Add time ticks in seconds
    spec_mag = torch.abs(planes['spectral']).squeeze().detach().cpu().numpy()
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4  # Assuming default hop_length = n_fft/4
    duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 7)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, duration_sec, 7)]
    
    # Get planes and detach tensors to handle parameters that require grad
    harm = planes['harmonic'].squeeze().detach().cpu().numpy()
    spatial = planes['spatial'].squeeze().detach().cpu().numpy() 
    psycho = planes['psychoacoustic'].squeeze().detach().cpu().numpy()
    
    # 1. Spectral magnitude (top left)
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
    
    # 2. Harmonic plane (log1p, 0-10000 Hz) (top middle)
    log_harm = np.log1p(harm)
    im2 = axs[0, 1].imshow(
        log_harm, 
        aspect='auto', 
        origin='lower',
        extent=[0, harm.shape[1], 0, 10000]  # Show up to 10000 Hz
    )
    harmonic_method = "Filterbank" if harm.max() < 1000 else "HPS"  # Heuristic to detect method
    axs[0, 1].set_title(f"Harmonic Structure (Log1p, {harmonic_method})")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_yticks(freq_ticks_full)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[0, 1].set_xticks(time_ticks)
    axs[0, 1].set_xticklabels(time_labels)
    axs[0, 1].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axs[0, 1])
    
    # 3. Spatial plane - Interaural Phase Difference (IPD) (top right)
    if spatial.shape[0] >= 1:
        ipd = spatial[0]
        im3 = axs[0, 2].imshow(
            ipd, 
            aspect='auto', 
            origin='lower',
            extent=[0, ipd.shape[1], 0, sample_rate/2],
            cmap='hsv',  # Circular colormap for phase values
            vmin=-np.pi, 
            vmax=np.pi
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
        axs[0, 2].text(0.5, 0.5, "IPD not available\n(mono audio)", 
                      ha='center', va='center', transform=axs[0, 2].transAxes)
        axs[0, 2].set_title("Spatial - Interaural Phase Difference")
    
    # 4. Harmonic plane (focused on 0-1000 Hz) (bottom left)
    im4 = axs[1, 0].imshow(
        harm, 
        aspect='auto', 
        origin='lower',
        extent=[0, harm.shape[1], 0, 1000],  # Limited to 0-1000 Hz
        vmax=np.percentile(harm, 99.5)  # Clip extreme values for better visualization
    )
    axs[1, 0].set_title("Harmonic Structure (Raw, 0-1000 Hz)")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_yticks(freq_ticks_low)
    axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks_low])
    axs[1, 0].set_xticks(time_ticks)
    axs[1, 0].set_xticklabels(time_labels)
    axs[1, 0].set_xlabel("Time (s)")
    plt.colorbar(im4, ax=axs[1, 0])
    
    # 5. Spatial plane - Energy Panning (bottom middle)
    if spatial.shape[0] >= 2:
        energy = spatial[1]
        im5 = axs[1, 1].imshow(
            energy, 
            aspect='auto', 
            origin='lower',
            extent=[0, energy.shape[1], 0, sample_rate/2],
            cmap='coolwarm',  # Blue-red colormap for left-right
            vmin=-1, 
            vmax=1
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
        axs[1, 1].text(0.5, 0.5, "Energy panning not available\n(mono audio)", 
                      ha='center', va='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title("Spatial - Energy Panning (L/R)")
    
    # 6. Psychoacoustic plane (bottom right)
    im6 = axs[1, 2].imshow(
        psycho, 
        aspect='auto', 
        origin='lower',
        extent=[0, psycho.shape[1], 0, sample_rate/2],
        vmin=0, 
        vmax=1
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

def main():
    parser = argparse.ArgumentParser(description="Analyze audio with Wav2Tensor")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds (default: 10s)")
    parser.add_argument("--method", type=str, default="hps", choices=["hps", "filterbank"], 
                        help="Harmonic plane method (default: hps)")
    parser.add_argument("--planes", type=str, default="all", 
                        help="Planes to include, comma-separated (default: 'all', options: 'spectral', 'harmonic', 'spatial', 'psychoacoustic')")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no-data", action="store_true", help="Skip tensor data logging")
    parser.add_argument("--output-dir", type=str, default="audio", help="Output directory (default: audio)")
    parser.add_argument("--prefix", type=str, default=None, help="Custom output file prefix")
    args = parser.parse_args()
    
    # Parse planes configuration
    include_planes = None
    if args.planes != "all":
        include_planes = [p.strip() for p in args.planes.split(",")]
        print(f"Using selective plane configuration: {include_planes}")
    
    try:
        # Run the analysis
        analyze_audio_segment(
            audio_file=args.audio_file,
            output_dir=args.output_dir,
            start_time=args.start,
            duration=args.duration,
            harmonic_method=args.method,
            include_planes=include_planes,
            visualize=not args.no_viz,
            save_tensor_data=not args.no_data,
            output_prefix=args.prefix
        )
        
    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 