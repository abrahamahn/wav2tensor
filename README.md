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

## Optimization Options

Wav2Tensor offers several optimization options for balancing quality, speed, and memory usage:

### 1. Selective Plane Configuration

Choose which planes to include based on your application needs:
- **Full**: All planes for maximum representation quality
- **Balanced**: Spectral and harmonic planes for music and general audio
- **Minimal**: Only spectral plane for fastest processing
- **Spatial Focus**: Spectral and spatial planes for stereo processing

### 2. Adaptive Frequency Resolution

Reduce memory footprint by using a frequency-adaptive resolution that allocates more bins to perceptually important regions:
- Higher resolution in bass region where it matters most
- Significant memory reduction (up to 90%) with minimal perceptual impact

### 3. Configurable Hop Length

Adjust time resolution to balance detail and efficiency:
- Smaller hop length (e.g., 256): More temporal detail
- Larger hop length (e.g., 512): ~50% memory reduction, faster processing

Our benchmarks show these optimizations can reduce memory usage by up to 92% with moderate quality impact.

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

### Optimized Usage

```python
import torch
from wav2tensor import Wav2TensorCore

# Create an audio tensor
waveform = torch.randn(1, 2, 22050)  # 1 second of stereo audio

# Initialize Wav2Tensor with optimized configuration
wav2tensor_optimized = Wav2TensorCore(
    sample_rate=22050,
    n_fft=1024,
    hop_length=512,                             # Increased hop length for efficiency
    use_adaptive_freq=True,                     # Higher resolution in bass region
    target_freq_bins=128,                       # Reduced frequency bins
    include_planes=['spectral', 'harmonic'],    # Only essential planes
    harmonic_method='hps',                      # Faster harmonic method
    use_half_precision=True                     # Use float16 for more efficiency
)

# Convert to optimized representation
tensor, planes = wav2tensor_optimized(waveform)
```

### Comparing Representations

Run the benchmark scripts to compare different configurations:

```bash
python benchmarks/run_benchmark.py path/to/your/audio.wav --planes minimal --adaptive --target_freq_bins 128
```

Options:
- `--planes`: 'default', 'all', 'minimal', or comma-separated list
- `--method`: 'hps' or 'filterbank' (default: 'hps')
- `--adaptive`: Enable adaptive frequency resolution
- `--target_freq_bins`: Number of frequency bins with adaptive resolution
- `--hop_length`: Hop length for STFT (default: 256)

### Model-based Comparison

To evaluate the performance of Wav2Tensor in actual models:

```bash
python examples/model_comparison.py
```

This script:
1. Creates a synthetic dataset for audio enhancement
2. Trains a U-Net model using different audio representations:
   - Raw waveform
   - Mel-spectrogram
   - Wav2Tensor (various configurations)
3. Evaluates performance using PESQ and STOI metrics
4. Generates comparison plots

### Example Scripts

Run the basic usage example to visualize the Wav2Tensor representation:

```bash
python -m examples.basic_usage
```

Or run with your own audio file:

```bash
python -m examples.basic_usage path/to/your/audio.wav
```

### Benchmark Results

Our benchmark results show the performance characteristics of different representations:

| Method          | Processing Time | Memory Usage | Output Shape      |
|-----------------|----------------:|-------------:|-------------------|
| Waveform        |     0.000095s   |    516.80 KB | [1, 2, 66150]     |
| Mel-Spectrogram |     0.000865s   |    161.88 KB | [1, 2, 1, 80, 259]|
| Wav2Tensor (Full)|     0.003455s   |   2076.05 KB | [1, 4, 513, 259]  |
| Wav2Tensor (Minimal)|     0.000570s   |    225.47 KB | [1, 4, 111, 130]  |

For more detailed benchmark results and optimization recommendations, see [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Performance Metrics [Planned]

While our benchmarks have measured computational efficiency, we are still validating the performance claims for downstream tasks. The `examples/model_comparison.py` script will evaluate Wav2Tensor against baselines using these metrics:

- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **SDR**: Signal-to-Distortion Ratio

Initial experiments with Wav2Tensor suggest promising improvements in these metrics, but comprehensive validation is still in progress. Results will be updated here as they become available.

## Project Structure

```
wav2tensor/
├── wav2tensor/         # Main package
│   ├── __init__.py     # Package exports
│   └── core.py         # Core implementation
├── examples/           # Example scripts
│   ├── basic_usage.py  # Basic usage demonstration
│   ├── model_comparison.py # Compare representations in models
│   └── evaluation.py   # Evaluation against baselines
├── benchmarks/         # Benchmarking tools
│   ├── processors/     # Audio representation processors
│   ├── configs/        # Configuration files
│   ├── runners/        # Benchmark runners
│   ├── visualizers/    # Results visualization 
│   ├── run_benchmark.py # Main benchmark script
│   ├── run_from_config.py # Config-based benchmark
│   └── RESULTS.md      # Benchmark result details
├── tests/              # Test suite
│   ├── unit/           # Unit tests for individual components
│   │   ├── planes/     # Tests for individual planes
│   ├── integration/    # Integration tests
│   ├── performance/    # Performance and benchmarking tests
│   ├── analysis/       # Analysis scripts and tools
│   ├── conftest.py     # Shared test fixtures
│   └── README.md       # Test documentation
├── run_tests.py        # Script to run tests with coverage
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