"""
Mel-spectrogram processor for benchmarking.

This module implements a processor for computing Mel-spectrograms with benchmarking capabilities.
"""

import torch
import torchaudio
from typing import Dict, Optional, Tuple

from benchmarks.processors.base import BaseProcessor


class MelSpectrogramProcessor(BaseProcessor):
    """Processor for Mel-spectrograms with performance tracking."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0
    ):
        """
        Initialize the Mel-spectrogram processor.
        
        Args:
            sample_rate: Audio sample rate (default: 22050Hz)
            n_fft: FFT size for STFT (default: 1024)
            hop_length: Hop length for STFT (default: 256)
            n_mels: Number of Mel bands (default: 80)
            f_min: Minimum frequency (default: 0.0)
            f_max: Maximum frequency (default: sample_rate/2)
            power: Power of the magnitude spectrogram (default: 2.0)
        """
        super().__init__(sample_rate=sample_rate, name="mel_spectrogram")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.power = power
        
        # Initialize the Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            norm='slaney',
            mel_scale='htk'
        )
    
    def process(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process an audio waveform into a Mel-spectrogram and measure performance.
        
        Args:
            waveform: Audio waveform tensor with shape [batch, channels, samples]
            
        Returns:
            mel_spec: Mel-spectrogram tensor with shape [batch, channels, n_mels, time]
            metrics: Dictionary with performance metrics
        """
        # Store batch dimension but process with channels
        batch_size = waveform.shape[0]
        channels = waveform.shape[1]
        
        # Process all channels of all batch items
        mel_specs = []
        for batch_idx in range(batch_size):
            batch_mels = []
            for channel_idx in range(channels):
                # Process each channel
                channel_waveform = waveform[batch_idx, channel_idx].unsqueeze(0)
                channel_mel = self.mel_transform(channel_waveform)  # [1, n_mels, time]
                batch_mels.append(channel_mel)
            
            # Stack channels if multichannel
            if channels > 1:
                batch_mel = torch.stack(batch_mels, dim=0)  # [channels, n_mels, time]
            else:
                batch_mel = batch_mels[0]  # [1, n_mels, time]
                
            mel_specs.append(batch_mel)
        
        # Stack batches
        if batch_size > 1:
            mel_spec = torch.stack(mel_specs, dim=0)  # [batch, channels, n_mels, time]
        else:
            mel_spec = mel_specs[0].unsqueeze(0)  # [1, channels, n_mels, time]
        
        # Calculate memory usage
        memory_bytes = mel_spec.element_size() * mel_spec.nelement()
        memory_kb = memory_bytes / 1024
        
        # Collect metrics
        metrics = {
            "memory_kb": memory_kb,
            "tensor_shape": list(mel_spec.shape),
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels
        }
        
        return mel_spec, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute Mel-spectrogram with benchmarking")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size (default: 1024)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length (default: 256)")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bands (default: 80)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: entire file)")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MelSpectrogramProcessor(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # Process audio file
    output, metrics = processor.process_file(
        audio_file=args.audio_file,
        segment_duration=args.duration,
        start_time=args.start
    )
    
    # Benchmark
    import torchaudio
    waveform, _ = torchaudio.load(args.audio_file)
    
    # Extract segment if requested
    if args.duration is not None:
        start_sample = int(args.start * args.sr)
        segment_samples = int(args.duration * args.sr)
        end_sample = min(start_sample + segment_samples, waveform.shape[1])
        waveform = waveform[:, start_sample:end_sample]
    
    # Benchmark
    metrics = processor.benchmark(waveform, n_runs=args.n_runs)
    
    # Print benchmark results
    print(f"Mel-spectrogram processor benchmark results ({args.n_runs} runs):")
    print(f"  Average processing time: {metrics['avg_process_time']:.6f} seconds")
    print(f"  Memory usage: {metrics['avg_memory_kb']:.2f} KB")
    print(f"  Shape: {metrics['tensor_shape']}") 