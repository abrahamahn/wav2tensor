# Wav2Tensor Examples

This directory contains example scripts for working with the Wav2Tensor audio representation library.

## Reconstruction Quality Test

The `reconstruction_test.py` script evaluates how well different audio representations (Wav2Tensor, mel-spectrogram, raw waveform) preserve audio information during conversion to and from the representation.

### Setup

First, generate test audio files:

```bash
python generate_test_audio.py --output test_audio
```

This creates various synthetic audio files with different characteristics (harmonics, transients, stereo content) to test the reconstruction capabilities.

### Running the Test

```bash
python reconstruction_test.py --audio test_audio --output results/reconstruction_test
```

This will:
1. Load each test audio file
2. Convert to various audio representations:
   - Raw waveform (identity)
   - STFT (complex spectrogram)
   - Mel-spectrogram
   - Wav2Tensor (full configuration)
   - Wav2Tensor (minimal configuration)
   - Wav2Tensor (balanced configuration)
3. Convert back to audio
4. Measure reconstruction quality using objective metrics (MSE, SNR, Log-Spectral Distance)
5. Generate comparison plots and save results

### Test Metrics

- **Mean Squared Error (MSE)**: Measures time-domain differences (lower is better)
- **Signal-to-Noise Ratio (SNR)**: Measures ratio of signal to reconstruction error (higher is better)
- **Log-Spectral Distance (LSD)**: Measures spectral differences (lower is better)

### Output

Results are saved to the specified output directory, including:
- Reconstructed audio files
- Waveform comparison plots
- Spectrogram difference plots
- JSON files with detailed metrics
- Summary plots comparing all representations

## Model Comparison

The `model_comparison.py` script evaluates the performance of different audio representations when used in standard deep learning models for audio enhancement tasks.

### Requirements

Before running, ensure you have installed the required dependencies:

```bash
pip install -r ../requirements.txt
```

### Running the Comparison

To run the full comparison:

```bash
python model_comparison.py
```

This will:
1. Create a synthetic dataset for audio enhancement
2. Train U-Net models using different audio representations:
   - Raw waveform
   - Mel-spectrogram
   - Wav2Tensor (full configuration)
   - Wav2Tensor (minimal configuration)
   - Wav2Tensor (balanced configuration)
3. Evaluate each model using MSE and SNR metrics
4. Generate comparison plots and save results to `results/model_comparison/`

### Customizing the Experiment

You can modify the script to customize:

- Model architecture
- Training parameters
- Dataset size and characteristics
- Audio representation configurations

### Output

The script generates several output files:

1. Individual JSON result files for each representation
2. A summary JSON file with all metrics
3. Comparison bar charts for MSE, SNR, and training time
4. Training and validation loss curves

All results are saved to the `results/model_comparison/` directory.

### Performance Metrics

The script evaluates the following metrics:

- **MSE**: Mean Squared Error (lower is better)
- **SNR**: Signal-to-Noise Ratio in dB (higher is better)
- **Training Time**: Time taken to train the model
- **Model Parameters**: Number of parameters in the model

### Example Output

```
===== Running experiment: Raw Waveform =====
...
Average MSE: 0.0854
Average SNR: 12.42

===== Running experiment: Mel-Spectrogram =====
...
Average MSE: 0.0523
Average SNR: 15.68

===== Running experiment: Wav2Tensor (Full) =====
...
Average MSE: 0.0346
Average SNR: 18.25
```

## Generate Test Audio

The `generate_test_audio.py` script creates synthetic audio files for testing the reconstruction quality of different audio representations.

```bash
python generate_test_audio.py --output test_audio --sr 22050 --duration 5.0
```

This generates the following test files:
- `sine_sweep.wav`: Logarithmic frequency sweep
- `harmonic_tones.wav`: Tone with harmonic overtones
- `noise_bursts.wav`: Periodic noise bursts
- `stereo_panning.wav`: Tone panning between left and right channels
- `complex_mix.wav`: Complex mix of tones, noise, and transients

## Basic Usage

The `basic_usage.py` script demonstrates how to use Wav2Tensor with different configurations.

To run with the default audio file:

```bash
python basic_usage.py
```

Or with your own audio file:

```bash
python basic_usage.py path/to/your/audio.wav
```

## Evaluation

The `evaluation.py` script compares Wav2Tensor against spectrogram-based baselines using established audio enhancement models.

```bash
python evaluation.py
```

This script uses real-world audio datasets for more comprehensive evaluation. 