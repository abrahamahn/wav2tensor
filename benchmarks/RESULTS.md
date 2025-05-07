# Benchmark Results Summary

This document summarizes the benchmark results for different audio representation methods in the wav2tensor project.

## Benchmarked Representations

1. **Raw Waveform** - The direct audio waveform data
2. **Mel-Spectrogram** - Standard mel-scaled spectrogram
3. **Wav2Tensor** - Multi-plane audio representation

## Test Conditions

- **Audio File**: `test_tone.wav` (1 second segment)
- **Sample Rate**: 22050 Hz
- **FFT Size**: 1024
- **Hop Length**: 256
- **Number of Runs**: 5 for each processor

## Default Configuration Results

Results with default configuration:

| Method          | Processing Time | Memory Usage | Output Shape      |
|-----------------|----------------:|-------------:|-------------------|
| Waveform        |     0.000095s   |    516.80 KB | [1, 2, 66150]     |
| Mel-Spectrogram |     0.000865s   |    161.88 KB | [1, 2, 1, 80, 259]|
| Wav2Tensor      |     0.003455s   |   2076.05 KB | [1, 4, 513, 259]  |

## Minimal Configuration Results

Results with minimal configuration (spectral plane only):

| Method          | Processing Time | Memory Usage | Output Shape      |
|-----------------|----------------:|-------------:|-------------------|
| Waveform        |     0.000121s   |    516.80 KB | [1, 2, 66150]     |
| Mel-Spectrogram |     0.000926s   |     81.25 KB | [1, 2, 1, 80, 130]|
| Wav2Tensor      |     0.000570s   |    225.47 KB | [1, 4, 111, 130]  |

## Observations

1. **Processing Time**:
   - Raw waveform processing is fastest, as expected (minimal processing)
   - Wav2Tensor with minimal configuration is competitive with mel-spectrogram
   - Full Wav2Tensor with all planes is ~4x slower than mel-spectrogram

2. **Memory Usage**:
   - Mel-spectrogram is most memory-efficient in default configuration
   - Full Wav2Tensor uses significant memory due to multiple planes
   - Minimal Wav2Tensor with adaptive frequency uses memory comparable to mel-spectrogram

3. **Tensor Shapes**:
   - Wav2Tensor output shape depends on included planes
   - Using adaptive frequency resolution significantly reduces the frequency dimension (from 513 to 111-187 bins)

## Recommendations

Based on these benchmark results:

1. **For Speed**:
   - Use raw waveform if minimal processing is needed
   - Use minimal Wav2Tensor (spectral plane only) for fast yet rich representation

2. **For Memory Efficiency**:
   - Mel-spectrogram is best for memory-constrained environments
   - Minimal Wav2Tensor with adaptive frequency offers a good tradeoff between memory and features

3. **For Feature Richness**:
   - Full Wav2Tensor provides the richest representation
   - Consider tradeoffs between processing time, memory usage, and feature richness

4. **Optimization Options**:
   - Increase hop length (e.g., from 256 to 512) to reduce time frames and memory usage
   - Use adaptive frequency resolution with target_freq_bins=128 for significant memory savings
   - Select only the planes needed for your specific application 