"""
Raw waveform processing with benchmarking.
This script provides functions to process raw waveforms from audio files and measure performance.
"""

import os
import time
import torch
import torchaudio
import numpy as np
from typing import Tuple, Dict, Optional

class WaveformProcessor:
    """
    Processor for raw waveforms with performance tracking.
    """
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
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.channels = channels
    
    def process(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process an audio waveform and measure performance.
        
        Args:
            waveform: Audio waveform tensor with shape [channels, samples]
            
        Returns:
            processed_waveform: Processed waveform tensor
            metrics: Dictionary with performance metrics
        """
        # Ensure batch dimension
        if len(waveform.shape) == 2:  # [channels, samples]
            waveform = waveform.unsqueeze(0)  # [batch, channels, samples]
        
        # Start timing
        start_time = time.time()
        
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
        
        # End timing
        process_time = time.time() - start_time
        
        # Calculate memory usage
        memory_bytes = processed_waveform.element_size() * processed_waveform.nelement()
        memory_kb = memory_bytes / 1024
        
        # Collect metrics
        metrics = {
            "process_time": process_time,
            "memory_kb": memory_kb,
            "tensor_shape": list(processed_waveform.shape)
        }
        
        return processed_waveform, metrics

    def process_file(self, audio_file: str, segment_duration: Optional[float] = None, start_time: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        """
        Process an audio file into a waveform and measure performance.
        
        Args:
            audio_file: Path to audio file
            segment_duration: Duration of segment to process in seconds (default: entire file)
            start_time: Start time in seconds (default: 0.0)
            
        Returns:
            waveform: Processed waveform tensor
            metrics: Dictionary with performance metrics
        """
        # Load audio file
        waveform, file_sr = torchaudio.load(audio_file)
        
        # Add metadata about the file
        file_metrics = {
            "original_sr": file_sr,
            "original_duration": waveform.shape[1] / file_sr,
            "original_channels": waveform.shape[0]
        }
        
        # Resample if needed
        if file_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(file_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Extract segment if requested
        if segment_duration is not None:
            start_sample = int(start_time * self.sample_rate)
            segment_samples = int(segment_duration * self.sample_rate)
            end_sample = min(start_sample + segment_samples, waveform.shape[1])
            waveform = waveform[:, start_sample:end_sample]
        
        # Process waveform
        processed_waveform, metrics = self.process(waveform)
        
        # Add file metadata to metrics
        metrics.update(file_metrics)
        
        return processed_waveform, metrics

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process waveform with benchmarking")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio to [-1, 1]")
    parser.add_argument("--channels", type=int, choices=[1, 2], help="Convert to mono (1) or stereo (2)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: entire file)")
    args = parser.parse_args()
    
    # Initialize processor
    processor = WaveformProcessor(
        sample_rate=args.sr,
        normalize=args.normalize,
        channels=args.channels
    )
    
    # Process audio file
    waveform, metrics = processor.process_file(
        audio_file=args.audio_file,
        segment_duration=args.duration,
        start_time=args.start
    )
    
    # Print performance metrics
    print(f"Processed waveform shape: {waveform.shape}")
    print(f"Original sample rate: {metrics['original_sr']} Hz")
    print(f"Original duration: {metrics['original_duration']:.2f} seconds")
    print(f"Original channels: {metrics['original_channels']}")
    print(f"Processing time: {metrics['process_time']:.4f} seconds")
    print(f"Memory usage: {metrics['memory_kb']:.2f} KB") 