# Wav2Tensor Benchmarks

This directory contains tools for benchmarking and comparing different audio representation methods:

1. Raw waveform
2. Mel-spectrogram
3. Wav2Tensor (full implementation)
4. Wav2TensorLite (optimized implementation)

## Directory Structure

- `benchmarks/` - Main directory
  - `processors/` - Audio representation processors
  - `configs/` - Configuration management
  - `runners/` - Benchmark execution
  - `visualizers/` - Results visualization (coming soon)
  - `results/` - Benchmark results storage

## Running Benchmarks

There are two ways to run benchmarks:

### 1. Using Command-Line Arguments

```bash
python benchmarks/run_benchmark.py test_tone.wav
```

Optional arguments:
- `--sr` - Sample rate (default: 22050)
- `--n_fft` - FFT size (default: 1024)
- `--hop_length` - Hop length (default: 256)
- `--n_mels` - Number of Mel bands (default: 80)
- `--planes` - Planes to include: 'default', 'all', 'minimal', or comma-separated list
- `--method` - Harmonic plane method: 'hps' or 'filterbank' (default: 'hps')
- `--bit_depth` - Bit depth for Wav2TensorLite quantization: 8 or 16 (default: 16)
- `--fusion` - Fusion method for Wav2TensorLite: 'concat', 'add', or 'learned' (default: 'concat')
- `--adaptive` - Use adaptive frequency resolution (default: False)
- `--target_freq_bins` - Number of frequency bins with adaptive frequency (default: 256)
- `--start` - Start time in seconds (default: 0.0)
- `--duration` - Duration in seconds (default: entire file)
- `--output_dir` - Directory to save benchmark results (default: 'benchmarks/results')
- `--n_runs` - Number of runs for each method (default: 5)

### 2. Using a Configuration File

```bash
python benchmarks/run_from_config.py benchmarks/configs/default_benchmark.yaml test_tone.wav
```

Optional arguments:
- `--output_dir` - Override output directory from config

## Configuration Files

Configuration files are in YAML format and allow you to specify all benchmark parameters. An example configuration:

```yaml
# Default benchmark configuration
sample_rate: 22050
n_fft: 1024
hop_length: 256
n_mels: 80
harmonic_method: hps
bit_depth: 16
fusion_method: concat
use_adaptive_freq: false
target_freq_bins: 256
include_planes:
  - spectral
  - harmonic
  - spatial
  - psychoacoustic
n_runs: 5
segment_duration: null  # Process entire file
start_time: 0.0
output_dir: benchmarks/results
```

## Benchmark Results

Benchmark results are saved to the specified output directory in JSON format. The results include:

- Processing time for each method
- Memory usage for each method
- Speedup factors between different methods
- Output tensor shapes
- Configuration parameters

You can use these results to compare the performance of different audio representation methods. 