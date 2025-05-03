"""
Basic usage example for Wav2Tensor.

This script demonstrates how to:
1. Load an audio file
2. Convert it to the Wav2Tensor representation
3. Visualize the different planes
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path to import wav2tensor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from wav2tensor import Wav2TensorCore

def plot_tensor_planes(planes, title="Wav2Tensor Representation", sample_rate=22050, n_fft=1024):
    """Plot the different planes of the Wav2Tensor representation."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Calculate frequency axis in Hz
    freqs = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
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
    plt.colorbar(im1, ax=axs[0, 0])
    
    # Add y-ticks at meaningful frequencies
    freq_ticks = [20, 100, 500, 1000, 2000, 5000, 10000]
    axs[0, 0].set_yticks(freq_ticks)
    axs[0, 0].set_yticklabels([f"{f}" for f in freq_ticks])
    
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
    plt.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_yticks(freq_ticks)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks])
    
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
        axs[1, 0].set_xlabel("Time frame")
        axs[1, 0].set_ylabel("Frequency (Hz)")
        plt.colorbar(im3, ax=axs[1, 0])
        axs[1, 0].set_yticks(freq_ticks)
        axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks])
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
    axs[1, 1].set_xlabel("Time frame")
    axs[1, 1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im4, ax=axs[1, 1])
    axs[1, 1].set_yticks(freq_ticks)
    axs[1, 1].set_yticklabels([f"{f}" for f in freq_ticks])
    
    plt.tight_layout()
    return fig

def main():
    # Check if a file path is provided as an argument
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Use a test tone if no file is provided
        print("No audio file provided, generating a test tone...")
        # Generate a test tone with multiple harmonics
        sample_rate = 22050
        duration = 3  # seconds
        t = torch.arange(0, duration, 1/sample_rate)
        
        # Create a tone with fundamental at 440 Hz and harmonics
        f0 = 440
        signal = torch.zeros(len(t))
        for i in range(1, 6):  # Add 5 harmonics
            signal += (1.0 / i) * torch.sin(2 * np.pi * i * f0 * t)
        
        # Add a simple amplitude envelope
        envelope = torch.ones_like(signal)
        envelope[:int(0.1 * sample_rate)] = torch.linspace(0, 1, int(0.1 * sample_rate))
        envelope[-int(0.1 * sample_rate):] = torch.linspace(1, 0, int(0.1 * sample_rate))
        signal = signal * envelope
        
        # Normalize
        signal = signal / torch.max(torch.abs(signal))
        
        # Add a second channel for stereo (with phase shift for spatial cues)
        signal_right = torch.sin(2 * np.pi * f0 * (t + 0.001))  # Phase shifted
        for i in range(2, 6):  # Add 5 harmonics
            signal_right += (1.0 / i) * torch.sin(2 * np.pi * i * f0 * (t + 0.001))
        signal_right = signal_right * envelope
        signal_right = signal_right / torch.max(torch.abs(signal_right))
        
        # Stack channels to make a stereo waveform [2, T]
        stereo_waveform = torch.stack([signal, signal_right])
        
        # For testing, let's save this waveform
        audio_path = "test_tone.wav"
        # torchaudio.save expects [channels, samples] format
        torchaudio.save(audio_path, stereo_waveform, sample_rate)
        print(f"Test tone saved to {audio_path}")
        
        # Prepare for Wav2Tensor - add batch dimension [B, C, T]
        waveform = stereo_waveform.unsqueeze(0)
    
    # If we're loading an actual file
    if audio_path and not 'waveform' in locals():
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Loaded audio: {audio_path}")
            print(f"Shape: {waveform.shape}, Sample rate: {sample_rate} Hz")
            
            # Add batch dimension if needed
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [B, C, T]
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return
    
    # Initialize Wav2Tensor encoder
    wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
    
    # Convert to Wav2Tensor representation
    tensor, planes = wav2tensor(waveform)
    print(f"Wav2Tensor representation shape: {tensor.shape}")
    
    # Get shapes of individual planes
    print("\nPlane shapes:")
    for name, plane in planes.items():
        print(f"  {name}: {plane.shape}")
    
    # Plot the different planes
    fig = plot_tensor_planes(
        planes, 
        title=f"Wav2Tensor: {os.path.basename(audio_path)}",
        sample_rate=sample_rate,
        n_fft=wav2tensor.n_fft
    )
    
    # Save the plot
    output_path = os.path.splitext(audio_path)[0] + "_planes.png"
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main() 