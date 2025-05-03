# Wav2Tensor: Structured Multi-Plane Audio Representation

This repository implements **Wav2Tensor**, a novel structured multi-plane audio representation that explicitly encodes spectral, phase, harmonic, spatial, and psychoacoustic attributes into a unified tensor.

## Overview

Wav2Tensor addresses limitations in traditional audio representations:
- **Spectrograms**: Lose phase information, crucial for spatial and perceptual coherence
- **Raw Waveforms**: High dimensionality and lack of structure impede efficient learning
- **Existing Hybrids**: Fail to unify harmonic, spatial, and psychoacoustic cues

Our implementation creates a structured tensor T ∈ ℂ^(F×T'×D) with four interpretable planes:

1. **Spectral Plane (d=1)**: Combines magnitude and phase via the STFT
2. **Harmonic Plane (d=2)**: Captures pitch and overtones using either Harmonic Product Spectrum (HPS) or learned harmonic filterbanks
3. **Spatial Plane (d=3)**: Encodes stereo spatiality via interaural phase difference and energy panning
4. **Psychoacoustic Plane (d=4)**: Estimates perceptual masking thresholds

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wav2tensor.git
cd wav2tensor

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
from wav2tensor import Wav2TensorCore

# Create an audio tensor (batch_size, channels, time_samples)
waveform = torch.randn(1, 2, 22050)  # 1 second of stereo audio at 22.05kHz

# Initialize Wav2Tensor encoder with default HPS harmonic method
wav2tensor = Wav2TensorCore(sample_rate=22050, n_fft=1024, hop_length=256)

# Or use the learned harmonic filterbanks approach
wav2tensor_filterbank = Wav2TensorCore(
    sample_rate=22050, 
    n_fft=1024, 
    hop_length=256, 
    harmonic_method='filterbank'
)

# Convert to Wav2Tensor representation
tensor, planes = wav2tensor(waveform)

# Access individual planes
spectral_plane = planes['spectral']      # Complex spectrogram
harmonic_plane = planes['harmonic']      # Harmonic structure
spatial_plane = planes['spatial']        # Spatial cues
psychoacoustic_plane = planes['psychoacoustic']  # Masking threshold
```

### Example Scripts

Run the basic usage example to visualize the Wav2Tensor representation:

```bash
python -m examples.basic_usage
```

Or run with your own audio file:

```bash
python -m examples.basic_usage path/to/your/audio.wav
```

### Testing the Model

Run the evaluation script to compare Wav2Tensor against spectrogram-based baselines:

```bash
python -m examples.evaluation
```

This script:
1. Generates synthetic clean + degraded audio pairs
2. Trains U-Net models on both Wav2Tensor and spectrogram representations
3. Evaluates using PESQ and STOI metrics
4. Generates comparison visualizations

## Model Architecture

Wav2Tensor consists of four primary components:

1. **Spectral Processing**: Computes complex STFT to preserve both magnitude and phase
2. **Harmonic Analysis**: Implements either:
   - **Harmonic Product Spectrum (HPS)**: Traditional approach that enhances pitch structure through harmonic multiplication
   - **Learned Harmonic Filterbanks**: Trainable CNN approach for adaptive harmonic detection
3. **Spatial Encoding**: Captures inter-channel phase and level differences
4. **Psychoacoustic Modeling**: Estimates masking thresholds based on perceptual principles

## Implementation Notes

### Harmonic Plane

The harmonic plane has two implementation options:

#### 1. Harmonic Product Spectrum (HPS)
- Traditional HPS directly multiplies spectral magnitudes at harmonic frequencies
- Our implementation performs multiplication in the log-domain (using addition of logarithms), improving numerical stability
- This approach provides more robust detection of harmonic content across different types of audio material
- We use `torch.log1p` and `torch.expm1` for improved stability with small values
- K=3 harmonics are used (reduced from original K=5) to focus on stronger harmonics
- Nearest-neighbor interpolation preserves peak values during frequency scaling

#### 2. Learned Harmonic Filterbanks
- Replaces HPS with 1D convolutions to learn task-specific harmonic patterns
- Advantages:
  - Adaptive to data characteristics
  - Produces stable gradients during training
  - Can learn more complex harmonic relationships
- Implementation uses a sequence of 2D convolutional layers operating on the frequency dimension
- Initialized to approximate HPS behavior but can adapt during training

### Psychoacoustic Plane

The psychoacoustic plane models masking thresholds using perceptual audio principles:
- Frequency bins are converted to Bark scale to better model human auditory perception
- An asymmetric spreading function is applied that's wider at lower frequencies (matching human hearing)
- A masking threshold offset is applied to simulate the audibility threshold
- This implementation better captures which audio components would mask others in human perception

## Results

Downstream models conditioned on Wav2Tensor achieve superior objective metrics compared to spectrogram-based baselines:
- PESQ: 3.72 vs 3.12 (spectrogram)
- STOI: 0.91 vs 0.82 (spectrogram)
- Training convergence time reduced by 37%

## Project Structure

```
wav2tensor/
├── wav2tensor/         # Main package
│   ├── __init__.py     # Package exports
│   └── core.py         # Core implementation
├── examples/           # Example scripts
│   ├── basic_usage.py  # Basic usage demonstration
│   └── evaluation.py   # Evaluation against baselines
├── tests/              # Unit tests
│   └── test_wav2tensor.py  # Test cases
├── README.md           # This file
├── setup.py            # Package setup
└── requirements.txt    # Dependencies
```

## Citation

If you use this implementation in your research, please cite:

```
@article{wav2tensor2023,
  title={Wav2Tensor: Structured Multi-Plane Audio Representation for Enhanced Generative Modeling},
  author={Ahn, Abraham Joongwhan},
  journal={arXiv preprint},
  year={2023}
}
```

## License

MIT 