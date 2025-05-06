"""
Script to compare spectral magnitude and psychoacoustic planes.
This will help verify that the psychoacoustic features are now more distinct
from the simple spectral magnitude.
"""
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor import Wav2TensorCore

def compare_planes(audio_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Loaded audio: {audio_path}")
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # [B, C, T]
    
    # Initialize Wav2Tensor encoder
    wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
    
    # Convert to Wav2Tensor representation
    _, planes = wav2tensor(waveform)
    
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
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # Calculate frequency axis in Hz
    sample_rate = wav2tensor.sample_rate
    n_fft = wav2tensor.n_fft
    
    # Add meaningful frequency ticks
    freq_ticks = [20, 100, 500, 1000, 2000, 5000, 10000]
    
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
    axs[2].set_xlabel("Time frame")
    axs[2].set_yticks(freq_ticks)
    axs[2].set_yticklabels([f"{f}" for f in freq_ticks])
    plt.colorbar(im3, ax=axs[2])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.splitext(audio_path)[0] + "_comparison.png"
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "audio/song1.mp3"
    
    compare_planes(audio_path) 