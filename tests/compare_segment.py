"""
Compare spectral magnitude and psychoacoustic planes for a specific audio segment.
This helps verify the distinction between the two planes in a shorter time span.
"""
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor import Wav2TensorCore

def compare_planes(audio_path, start_time=0.0, duration=30.0):
    """Compare spectral magnitude and psychoacoustic planes for a segment."""
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Loaded audio: {audio_path}")
    print(f"Original shape: {waveform.shape}, Sample rate: {sample_rate} Hz")
    
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
    
    # Initialize Wav2Tensor encoder
    wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
    
    # Convert to Wav2Tensor representation
    _, planes = wav2tensor(segment)
    
    # Get spectral magnitude and psychoacoustic planes
    spec_mag = torch.abs(planes['spectral']).squeeze().cpu().numpy()
    psycho = planes['psychoacoustic'].squeeze().cpu().numpy()
    
    # Calculate difference
    diff = np.abs(spec_mag / np.max(spec_mag) - psycho)
    
    # Calculate correlation
    correlation = np.corrcoef(
        spec_mag.reshape(-1), 
        psycho.reshape(-1)
    )[0, 1]
    
    print(f"Correlation between spectral magnitude and psychoacoustic planes: {correlation:.4f}")
    print(f"Mean absolute difference: {np.mean(diff):.4f}")
    
    # Create visualization
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    
    # Calculate frequency axis in Hz
    n_fft = wav2tensor.n_fft
    
    # Add meaningful frequency ticks
    freq_ticks = [20, 100, 500, 1000, 2000, 5000, 10000]
    
    # Add time ticks in seconds
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4  # Assuming default hop_length = n_fft/4
    segment_duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 7)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, segment_duration_sec, 7)]
    
    # Plot spectral magnitude
    im1 = axs[0].imshow(
        10 * np.log10(spec_mag + 1e-8), 
        aspect='auto', 
        origin='lower',
        extent=[0, spec_mag.shape[1], 0, sample_rate/2]
    )
    axs[0].set_title("Spectral Magnitude (dB)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_yticks(freq_ticks)
    axs[0].set_yticklabels([f"{f}" for f in freq_ticks])
    axs[0].set_xticks(time_ticks)
    axs[0].set_xticklabels(time_labels)
    axs[0].set_xlabel("Time (s)")
    plt.colorbar(im1, ax=axs[0])
    
    # Plot psychoacoustic features
    im2 = axs[1].imshow(
        psycho, 
        aspect='auto', 
        origin='lower',
        extent=[0, psycho.shape[1], 0, sample_rate/2]
    )
    axs[1].set_title("Psychoacoustic Features (Masking Threshold)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_yticks(freq_ticks)
    axs[1].set_yticklabels([f"{f}" for f in freq_ticks])
    axs[1].set_xticks(time_ticks)
    axs[1].set_xticklabels(time_labels)
    axs[1].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axs[1])
    
    # Plot difference
    im3 = axs[2].imshow(
        diff, 
        aspect='auto', 
        origin='lower', 
        cmap='viridis',
        extent=[0, diff.shape[1], 0, sample_rate/2]
    )
    axs[2].set_title("Absolute Difference")
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_yticks(freq_ticks)
    axs[2].set_yticklabels([f"{f}" for f in freq_ticks])
    axs[2].set_xticks(time_ticks)
    axs[2].set_xticklabels(time_labels)
    plt.colorbar(im3, ax=axs[2])
    
    # Adjust layout
    plt.tight_layout()
    
    # Generate output path
    base_name = os.path.splitext(audio_path)[0]
    segment_info = f"_{int(start_time)}s_to_{int(start_time + segment_duration)}s"
    output_path = f"{base_name}{segment_info}_comparison.png"
    
    # Set overall title
    fig.suptitle(f"Comparison: {os.path.basename(audio_path)} ({start_time:.1f}s to {start_time + segment_duration:.1f}s)", fontsize=16)
    plt.subplots_adjust(top=0.95)  # Adjust to make room for the title
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare planes for an audio segment")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds to analyze (default: 30s)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    args = parser.parse_args()
    
    try:
        compare_planes(args.audio_file, args.start, args.duration)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 