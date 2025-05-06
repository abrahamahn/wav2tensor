"""
Raw waveform processor for benchmarking.

This module implements a processor for raw audio waveform with benchmarking capabilities.
"""

import torch
from typing import Dict, Optional, Tuple, Any

from benchmarks.processors.base import BaseProcessor


class WaveformProcessor(BaseProcessor):
    """Processor for raw waveforms with performance tracking."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        normalize: bool = True,
        channels: Optional[int] = None  # Number of channels to convert to (e.g., mono=1, stereo=2)
    ):
        """
        Initialize the waveform processor.
        
        Args:
            sample_rate: Target audio sample rate (default: 22050Hz)
            normalize: Whether to normalize audio to [-1, 1] range (default: True)
            channels: Number of channels to convert to (default: None, keeps original)
        """
        super().__init__(sample_rate=sample_rate, name="waveform")
        self.normalize = normalize
        self.channels = channels
    
    def process(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process an audio waveform and measure performance.
        
        Args:
            waveform: Audio waveform tensor with shape [batch, channels, samples]
            
        Returns:
            processed_waveform: Processed waveform tensor
            metrics: Dictionary with performance metrics
        """
        # Clone to avoid modifying the input
        processed_waveform = waveform.clone()
        
        # Simple normalization to [-1, 1] range if requested
        if self.normalize:
            for batch_idx in range(processed_waveform.shape[0]):
                for channel_idx in range(processed_waveform.shape[1]):
                    channel_data = processed_waveform[batch_idx, channel_idx]
                    max_val = torch.max(torch.abs(channel_data))
                    if max_val > 0:  # Avoid division by zero
                        processed_waveform[batch_idx, channel_idx] = channel_data / max_val
        
        # Convert channels if needed
        if self.channels is not None:
            current_channels = processed_waveform.shape[1]
            
            if current_channels == 1 and self.channels == 2:
                # Mono to stereo: duplicate channel
                processed_waveform = torch.cat([processed_waveform, processed_waveform], dim=1)
            
            elif current_channels == 2 and self.channels == 1:
                # Stereo to mono: average channels
                processed_waveform = torch.mean(processed_waveform, dim=1, keepdim=True)
        
        # Calculate memory usage
        memory_bytes = processed_waveform.element_size() * processed_waveform.nelement()
        memory_kb = memory_bytes / 1024
        
        # Collect metrics
        metrics = {
            "memory_kb": memory_kb,
            "tensor_shape": list(processed_waveform.shape)
        }
        
        return processed_waveform, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process waveform with benchmarking")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio to [-1, 1]")
    parser.add_argument("--channels", type=int, choices=[1, 2], help="Convert to mono (1) or stereo (2)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: entire file)")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = WaveformProcessor(
        sample_rate=args.sr,
        normalize=args.normalize,
        channels=args.channels
    )
    
    # Load audio file
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
    print(f"Waveform processor benchmark results ({args.n_runs} runs):")
    print(f"  Average processing time: {metrics['avg_process_time']:.6f} seconds")
    print(f"  Memory usage: {metrics['avg_memory_kb']:.2f} KB")
    print(f"  Shape: {metrics['tensor_shape']}") 