"""
Audio processors for benchmarking.

This module contains processors for different audio representations:
- Waveform: Raw audio waveform processing
- MelSpectrogram: Mel-spectrogram transformation
- Wav2TensorProcessor: Full Wav2Tensor representation
- Wav2TensorLiteProcessor: Optimized Wav2Tensor representation
"""

from benchmarks.processors.waveform import WaveformProcessor
from benchmarks.processors.mel_spectrogram import MelSpectrogramProcessor
from benchmarks.processors.wav2tensor import Wav2TensorProcessor
from benchmarks.processors.wav2tensor_lite import Wav2TensorLiteProcessor

__all__ = [
    'WaveformProcessor', 
    'MelSpectrogramProcessor', 
    'Wav2TensorProcessor',
    'Wav2TensorLiteProcessor'
] 