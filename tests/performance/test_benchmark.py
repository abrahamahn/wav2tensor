"""
Performance tests for Wav2Tensor.
"""

import pytest
import torch
import numpy as np
import time
import os


class TestWav2TensorPerformance:
    """Test class for Wav2Tensor performance."""
    
    def test_inference_speed(self, wav2tensor_default, sample_audio_waveform):
        """Test the inference speed of Wav2Tensor."""
        # Warm-up run
        _, _ = wav2tensor_default(sample_audio_waveform)
        
        # Benchmark run
        n_runs = 5
        times = []
        
        for _ in range(n_runs):
            start_time = time.time()
            _, _ = wav2tensor_default(sample_audio_waveform)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Print results
        print(f"Wav2TensorCore inference time (seconds):")
        print(f"  Average: {avg_time:.4f}")
        print(f"  Min: {min_time:.4f}")
        print(f"  Max: {max_time:.4f}")
        
        # We don't have a strict performance requirement, but ensure it completes reasonably
        assert avg_time < 1.0, "Inference should be reasonably fast"
    
    def test_adaptive_vs_standard_performance(self, sample_audio_waveform):
        """Compare performance between adaptive and standard frequency resolution."""
        from wav2tensor import Wav2TensorCore
        
        # Create models
        standard_model = Wav2TensorCore(use_adaptive_freq=False)
        adaptive_model = Wav2TensorCore(use_adaptive_freq=True, target_freq_bins=128)
        
        # Warm-up runs
        _, _ = standard_model(sample_audio_waveform)
        _, _ = adaptive_model(sample_audio_waveform)
        
        # Benchmark
        n_runs = 5
        standard_times = []
        adaptive_times = []
        
        for _ in range(n_runs):
            # Standard model
            start_time = time.time()
            _, _ = standard_model(sample_audio_waveform)
            end_time = time.time()
            standard_times.append(end_time - start_time)
            
            # Adaptive model
            start_time = time.time()
            _, _ = adaptive_model(sample_audio_waveform)
            end_time = time.time()
            adaptive_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_standard = np.mean(standard_times)
        avg_adaptive = np.mean(adaptive_times)
        speedup = avg_standard / avg_adaptive
        
        # Print results
        print(f"Performance comparison:")
        print(f"  Standard: {avg_standard:.4f} seconds")
        print(f"  Adaptive: {avg_adaptive:.4f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Adaptive should generally be faster
        assert speedup >= 0.9, "Adaptive resolution should not be significantly slower"
    
    def test_memory_usage(self, sample_audio_waveform):
        """Test the memory usage of Wav2Tensor."""
        from wav2tensor import Wav2TensorCore
        
        # Memory measurement is tricky in Python, we'll use a simple approximation
        # based on the size of returned tensors
        
        # Create models
        standard_model = Wav2TensorCore(use_adaptive_freq=False)
        adaptive_model = Wav2TensorCore(use_adaptive_freq=True, target_freq_bins=128)
        
        # Process with both models
        tensor_standard, _ = standard_model(sample_audio_waveform)
        tensor_adaptive, _ = adaptive_model(sample_audio_waveform)
        
        # Calculate memory usage of the output tensors in MB
        memory_standard = tensor_standard.element_size() * tensor_standard.numel() / (1024 * 1024)
        memory_adaptive = tensor_adaptive.element_size() * tensor_adaptive.numel() / (1024 * 1024)
        
        # Print results
        print(f"Memory usage (MB):")
        print(f"  Standard: {memory_standard:.4f}")
        print(f"  Adaptive: {memory_adaptive:.4f}")
        print(f"  Reduction: {(memory_standard - memory_adaptive) / memory_standard * 100:.1f}%")
        
        # Adaptive should use less memory
        assert memory_adaptive <= memory_standard, "Adaptive should use less memory than standard"
    
    def test_batch_size_scaling(self):
        """Test how performance scales with batch size."""
        from wav2tensor import Wav2TensorCore
        
        # Create model
        model = Wav2TensorCore()
        
        # Create a simple waveform
        sample_rate = 22050
        t = torch.arange(0, 1.0, 1/sample_rate)
        waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # [C, T]
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            # Create batched input
            batch = torch.stack([waveform] * batch_size, dim=0)  # [B, C, T]
            
            # Warm-up
            _, _ = model(batch)
            
            # Benchmark
            n_runs = 3
            batch_times = []
            
            for _ in range(n_runs):
                start_time = time.time()
                _, _ = model(batch)
                end_time = time.time()
                batch_times.append(end_time - start_time)
            
            avg_time = np.mean(batch_times)
            times.append(avg_time)
        
        # Print results
        print(f"Batch size scaling:")
        for batch_size, avg_time in zip(batch_sizes, times):
            print(f"  Batch {batch_size}: {avg_time:.4f} seconds")
        
        # Calculate per-sample times to check linear scaling
        per_sample_times = [t / bs for t, bs in zip(times, batch_sizes)]
        
        # Allow for some overhead, but should roughly scale linearly
        # We're looking for no dramatic increases in per-sample processing time
        max_increase = max([per_sample_times[i+1] / per_sample_times[i] 
                            for i in range(len(per_sample_times) - 1)])
        print(f"  Max per-sample time increase factor: {max_increase:.2f}x")
        
        # The per-sample time shouldn't increase dramatically with batch size
        assert max_increase < 1.5, "Processing time per sample should scale reasonably with batch size" 