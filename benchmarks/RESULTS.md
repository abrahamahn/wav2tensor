# Benchmark Results Summary

This document summarizes the benchmark results for different audio representation methods in the wav2tensor project.

## Benchmarked Representations

1. **Raw Waveform** - The direct audio waveform data
2. **Mel-Spectrogram** - Standard mel-scaled spectrogram
3. **Wav2TensorLite** - Optimized multi-plane audio representation

## Test Conditions

- **Audio File**: `test_tone.wav` (1 second segment)
- **Sample Rate**: 22050 Hz
- **FFT Size**: 1024
- **Hop Length**: 256
- **Number of Runs**: 3 for each processor

## Default Configuration Results

Results with default configuration:

| Method          | Processing Time | Memory Usage | Output Shape      |
|-----------------|----------------:|-------------:|-------------------|
| Waveform        |     0.000102s   |    172.27 KB | [1, 2, 22050]     |
| Mel-Spectrogram |     0.000800s   |     54.38 KB | [1, 2, 80, 87]    |
| Wav2TensorLite  |     0.001946s   |    871.70 KB | [1, 5, 513, 87]   |

## Minimal Configuration Results

Results with minimal configuration (spectral plane only, 8-bit depth):

| Method          | Processing Time | Memory Usage | Output Shape      |
|-----------------|----------------:|-------------:|-------------------|
| Waveform        |     0.000101s   |    172.27 KB | [1, 2, 22050]     |
| Mel-Spectrogram |     0.000665s   |     54.38 KB | [1, 2, 80, 87]    |
| Wav2TensorLite  |     0.000762s   |    348.68 KB | [1, 2, 513, 87]   |

## Observations

1. **Processing Time**:
   - Raw waveform processing is fastest, as expected (minimal processing)
   - Wav2TensorLite with minimal configuration is only slightly slower than mel-spectrogram
   - Full Wav2TensorLite with all planes is ~2.4x slower than mel-spectrogram

2. **Memory Usage**:
   - Mel-spectrogram is most memory-efficient
   - Full Wav2TensorLite uses the most memory due to multiple planes
   - Minimal Wav2TensorLite with 8-bit depth reduces memory by ~60%

3. **Tensor Shapes**:
   - Wav2TensorLite output shape depends on included planes
   - More planes = higher channel dimension (5 channels for default config vs. 2 for minimal)

## Recommendations

Based on these benchmark results:

1. **For Speed**:
   - Use raw waveform if minimal processing is needed
   - Use minimal Wav2TensorLite with 8-bit depth for fast yet rich representation

2. **For Memory Efficiency**:
   - Mel-spectrogram is best for memory-constrained environments
   - Minimal Wav2TensorLite offers a good tradeoff between memory and features

3. **For Feature Richness**:
   - Full Wav2TensorLite provides the richest representation
   - Consider tradeoffs between processing time, memory usage, and feature richness 