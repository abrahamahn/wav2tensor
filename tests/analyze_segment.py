"""
Analyze just the first 30 seconds of an audio file with Wav2Tensor.
This allows for more detailed visualization of a shorter time span.
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

def plot_tensor_planes(planes, title="Wav2Tensor Representation", sample_rate=22050, n_fft=1024):
    """Plot the different planes of the Wav2Tensor representation."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    # Calculate frequency axis in Hz
    freqs = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
    # Add meaningful frequency ticks
    freq_ticks = [20, 100, 500, 1000, 2000, 5000, 10000]
    
    # Plot spectral magnitude
    spec_mag = torch.abs(planes['spectral']).squeeze().cpu().numpy()
    im1 = axs[0, 0].imshow(
        10 * np.log10(spec_mag + 1e-8), 
        aspect='auto', 
        origin='lower',
        extent=[0, spec_mag.shape[1], 0, sample_rate/2]
    )
    axs[0, 0].set_title("Spectral Magnitude (dB)")
    axs[0, 0].set_ylabel("Frequency (Hz)")
    axs[0, 0].set_yticks(freq_ticks)
    axs[0, 0].set_yticklabels([f"{f}" for f in freq_ticks])
    plt.colorbar(im1, ax=axs[0, 0])
    
    # Add time ticks in seconds
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4  # Assuming default hop_length = n_fft/4
    duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 7)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, duration_sec, 7)]
    axs[0, 0].set_xticks(time_ticks)
    axs[0, 0].set_xticklabels(time_labels)
    axs[0, 0].set_xlabel("Time (s)")
    
    # Plot harmonic structure
    harm = planes['harmonic'].squeeze().cpu().numpy()
    im2 = axs[0, 1].imshow(
        harm, 
        aspect='auto', 
        origin='lower',
        extent=[0, harm.shape[1], 0, sample_rate/2]
    )
    axs[0, 1].set_title("Harmonic Structure")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_yticks(freq_ticks)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks])
    axs[0, 1].set_xticks(time_ticks)
    axs[0, 1].set_xticklabels(time_labels)
    axs[0, 1].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axs[0, 1])
    
    # Plot spatial features (first channel - IPD)
    if planes['spatial'].shape[1] >= 2:
        spat = planes['spatial'][:, 0].squeeze().cpu().numpy()
        im3 = axs[1, 0].imshow(
            spat, 
            aspect='auto', 
            origin='lower', 
            cmap='coolwarm',
            extent=[0, spat.shape[1], 0, sample_rate/2]
        )
        axs[1, 0].set_title("Spatial Features (IPD)")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Frequency (Hz)")
        axs[1, 0].set_yticks(freq_ticks)
        axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks])
        axs[1, 0].set_xticks(time_ticks)
        axs[1, 0].set_xticklabels(time_labels)
        plt.colorbar(im3, ax=axs[1, 0])
    else:
        axs[1, 0].text(0.5, 0.5, "No spatial features\n(mono audio)", 
                     horizontalalignment='center', verticalalignment='center')
        axs[1, 0].set_title("Spatial Features")
    
    # Plot psychoacoustic features
    psych = planes['psychoacoustic'].squeeze().cpu().numpy()
    im4 = axs[1, 1].imshow(
        psych, 
        aspect='auto', 
        origin='lower',
        extent=[0, psych.shape[1], 0, sample_rate/2]
    )
    axs[1, 1].set_title("Psychoacoustic Features")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Frequency (Hz)")
    axs[1, 1].set_yticks(freq_ticks)
    axs[1, 1].set_yticklabels([f"{f}" for f in freq_ticks])
    axs[1, 1].set_xticks(time_ticks)
    axs[1, 1].set_xticklabels(time_labels)
    plt.colorbar(im4, ax=axs[1, 1])
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze a segment of an audio file with Wav2Tensor")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds to analyze (default: 30s)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    args = parser.parse_args()
    
    # Load the audio file
    try:
        waveform, sample_rate = torchaudio.load(args.audio_file)
        print(f"Loaded audio: {args.audio_file}")
        print(f"Original shape: {waveform.shape}, Sample rate: {sample_rate} Hz")
        
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
        print(f"Wav2Tensor representation shape: {tensor.shape}")
        
        # Get shapes of individual planes
        print("\nPlane shapes:")
        for name, plane in planes.items():
            print(f"  {name}: {plane.shape}")
        
        # Generate output path
        base_name = os.path.splitext(args.audio_file)[0]
        segment_info = f"_{int(args.start)}s_to_{int(args.start + segment_duration)}s"
        output_path = f"{base_name}{segment_info}_planes.png"
        
        # Plot the different planes
        title = f"Wav2Tensor: {os.path.basename(args.audio_file)} ({args.start:.1f}s to {args.start + segment_duration:.1f}s)"
        fig = plot_tensor_planes(
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