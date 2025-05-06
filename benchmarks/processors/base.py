"""
Base audio processor class for benchmarking.

This module provides a base class for audio processors that defines
the common interface and utility methods.
"""

import os
import time
import torch
import torchaudio
import numpy as np
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Base class for audio processors with benchmarking capabilities."""
    
    def __init__(self, sample_rate: int = 22050, name: str = "base"):
        """
        Initialize the base processor.
        
        Args:
            sample_rate: Target audio sample rate (default: 22050Hz)
            name: Processor name for benchmark results (default: "base")
        """
        self.sample_rate = sample_rate
        self.name = name
    
    @abstractmethod
    def process(self, waveform: torch.Tensor) -> Tuple[Any, Dict]:
        """
        Process an audio waveform and measure performance.
        
        Args:
            waveform: Audio waveform tensor with shape [batch, channels, samples]
            
        Returns:
            processed_output: Processed output tensor or data structure
            metrics: Dictionary with performance metrics
        """
        pass
    
    def process_file(self, audio_file: str, segment_duration: Optional[float] = None, start_time: float = 0.0) -> Tuple[Any, Dict]:
        """
        Process an audio file and measure performance.
        
        Args:
            audio_file: Path to audio file
            segment_duration: Duration of segment to process in seconds (default: entire file)
            start_time: Start time in seconds (default: 0.0)
            
        Returns:
            processed_output: Processed output tensor or data structure
            metrics: Dictionary with performance metrics
            
        Raises:
            FileNotFoundError: If the audio file does not exist
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load audio file
        waveform, file_sr = torchaudio.load(audio_file)
        
        # Add metadata about the file
        file_metrics = {
            "original_sr": file_sr,
            "original_duration": waveform.shape[1] / file_sr,
            "original_channels": waveform.shape[0],
            "file_path": audio_file
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
        
        # Ensure the waveform has batch dimension
        if len(waveform.shape) == 2:  # [channels, samples]
            waveform = waveform.unsqueeze(0)  # [batch, channels, samples]
        
        # Process waveform
        processed_output, metrics = self.process(waveform)
        
        # Add file metadata to metrics
        metrics.update(file_metrics)
        
        # Add processor identifier
        metrics["method"] = self.name
        
        return processed_output, metrics
    
    def benchmark(self, waveform: torch.Tensor, n_runs: int = 5) -> Dict:
        """
        Benchmark the processor on a waveform multiple times.
        
        Args:
            waveform: Audio waveform tensor
            n_runs: Number of benchmark runs (default: 5)
            
        Returns:
            metrics: Dictionary with average performance metrics
        """
        # Ensure the waveform has batch dimension
        if len(waveform.shape) == 2:  # [channels, samples]
            waveform = waveform.unsqueeze(0)  # [batch, channels, samples]
        
        # Run multiple times to get average metrics
        all_metrics = []
        
        for i in range(n_runs):
            # Run a warmup if it's the first run
            if i == 0:
                _, _ = self.process(waveform)
            
            # Measure processing time
            start_time = time.time()
            processed_output, metrics = self.process(waveform)
            process_time = time.time() - start_time
            
            # Override process time with more accurate measurement
            metrics["process_time"] = process_time
            
            # Add run index
            metrics["run_idx"] = i
            
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        # Add method identifier
        avg_metrics["method"] = self.name
        
        return avg_metrics
    
    def _aggregate_metrics(self, metrics_list: list) -> Dict:
        """
        Aggregate metrics from multiple runs.
        
        Args:
            metrics_list: List of metrics dictionaries from multiple runs
            
        Returns:
            aggregate_metrics: Dictionary with aggregated metrics
        """
        # Initialize aggregate metrics
        aggregate_metrics = {"runs": metrics_list}
        
        # Extract numeric metrics for averaging
        numeric_metrics = ["process_time", "memory_kb"]
        
        # Compute averages for numeric metrics
        for metric in numeric_metrics:
            if all(metric in m for m in metrics_list):
                values = [m[metric] for m in metrics_list]
                aggregate_metrics[f"avg_{metric}"] = sum(values) / len(values)
                aggregate_metrics[f"min_{metric}"] = min(values)
                aggregate_metrics[f"max_{metric}"] = max(values)
        
        # Include non-numeric metadata from the first run
        for key, value in metrics_list[0].items():
            if key not in numeric_metrics + ["run_idx"]:
                aggregate_metrics[key] = value
        
        return aggregate_metrics 