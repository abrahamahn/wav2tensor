"""
Shared fixtures for Wav2Tensor tests.
"""

import os
import sys
import pytest
import torch
import numpy as np
import torchaudio

# Add parent directory to path to import wav2tensor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from wav2tensor import Wav2TensorCore

@pytest.fixture
def wav2tensor_default():
    """Create a default Wav2Tensor instance for testing."""
    return Wav2TensorCore(sample_rate=22050, n_fft=1024, hop_length=256)

@pytest.fixture
def wav2tensor_with_adaptive_freq():
    """Create a Wav2Tensor instance with adaptive frequency for testing."""
    return Wav2TensorCore(
        sample_rate=22050, 
        n_fft=1024, 
        hop_length=256, 
        use_adaptive_freq=True, 
        target_freq_bins=128
    )

@pytest.fixture
def wav2tensor_filterbank():
    """Create a Wav2Tensor instance with filterbank for harmonic plane."""
    return Wav2TensorCore(
        sample_rate=22050, 
        n_fft=1024, 
        hop_length=256, 
        harmonic_method='filterbank'
    )

@pytest.fixture
def mono_waveform():
    """Create a simple mono test waveform."""
    # Create a 1-second sine wave at 440 Hz
    sample_rate = 22050
    t = torch.arange(0, 1.0, 1/sample_rate)
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [B, C, T]
    return waveform

@pytest.fixture
def stereo_waveform():
    """Create a simple stereo test waveform."""
    # Create a 1-second stereo signal with phase difference
    sample_rate = 22050
    t = torch.arange(0, 1.0, 1/sample_rate)
    left = torch.sin(2 * np.pi * 440 * t)
    right = torch.sin(2 * np.pi * 440 * (t + 0.001))  # Phase shifted
    waveform = torch.stack([left, right], dim=0).unsqueeze(0)  # [B, C, T]
    return waveform

@pytest.fixture
def harmonic_waveform():
    """Create a test waveform with strong harmonic content."""
    # Create a signal with strong harmonics
    sample_rate = 22050
    t = torch.arange(0, 1.0, 1/sample_rate)
    f0 = 440  # Hz
    
    # Create waveform with multiple harmonics
    waveform = torch.zeros_like(t)
    for i in range(1, 4):  # Add 3 harmonics
        waveform += (1/i) * torch.sin(2 * np.pi * i * f0 * t)
    
    # Normalize and reshape for processing
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = waveform.unsqueeze(0).unsqueeze(0)  # [B, C, T]
    return waveform

@pytest.fixture
def sample_audio_path():
    """Return path to a sample audio file."""
    # Using the test_tone.wav in the project root by default
    audio_path = os.path.join(parent_dir, "test_tone.wav")
    if os.path.exists(audio_path):
        return audio_path
    
    # Fallback to audio directory
    audio_dir = os.path.join(parent_dir, "audio")
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
        if audio_files:
            return os.path.join(audio_dir, audio_files[0])
    
    # If no files found, we'll skip tests that require this fixture
    pytest.skip("No sample audio files found for testing")

@pytest.fixture
def sample_audio_waveform(sample_audio_path):
    """Load a sample audio file as a waveform tensor."""
    waveform, _ = torchaudio.load(sample_audio_path)
    return waveform.unsqueeze(0)  # Add batch dimension [B, C, T] 