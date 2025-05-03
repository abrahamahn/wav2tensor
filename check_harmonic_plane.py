"""
Quick script to verify the harmonic plane is not all zeros after our fix.
"""
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor import Wav2TensorCore

def main():
    # Load the audio file
    audio_path = "audio/song1.mp3"
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Loaded audio: {audio_path}")
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # [B, C, T]
    
    # Initialize Wav2Tensor encoder
    wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
    
    # Convert to Wav2Tensor representation
    _, planes = wav2tensor(waveform)
    
    # Get harmonic plane
    harmonic_plane = planes['harmonic'].squeeze().cpu().numpy()
    
    # Check statistics
    print(f"Harmonic plane shape: {harmonic_plane.shape}")
    print(f"Harmonic plane min: {np.min(harmonic_plane)}")
    print(f"Harmonic plane max: {np.max(harmonic_plane)}")
    print(f"Harmonic plane mean: {np.mean(harmonic_plane)}")
    print(f"Number of zeros: {np.sum(harmonic_plane == 0)} out of {harmonic_plane.size}")
    print(f"Percentage of zeros: {100 * np.sum(harmonic_plane == 0) / harmonic_plane.size:.2f}%")
    
    # Visualize part of the harmonic plane
    plt.figure(figsize=(10, 6))
    plt.imshow(harmonic_plane[:, :1000], aspect='auto', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title('Harmonic Plane (First 1000 frames)')
    plt.xlabel('Time frame')
    plt.ylabel('Frequency bin')
    
    # Save the plot
    output_path = os.path.splitext(audio_path)[0] + "_harmonic.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main() 