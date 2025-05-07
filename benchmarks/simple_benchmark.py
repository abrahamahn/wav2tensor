#!/usr/bin/env python3
"""
Simple benchmark runner for Wav2Tensor.

A standalone script for comparing different audio representation methods.
"""

import os
import sys
import time
import json
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply monkey patch to fix Wav2TensorCore._ensure_4d issue
from benchmarks.monkey_patch import apply_patches
apply_patches()

# Basic processors for benchmarking
from benchmarks.processors.waveform import WaveformProcessor
from benchmarks.processors.mel_spectrogram import MelSpectrogramProcessor
from benchmarks.processors.wav2tensor import Wav2TensorProcessor

def load_audio(audio_file, sample_rate=22050, duration=None, start_time=0.0):
    """Load audio file and prepare for benchmarking."""
    waveform, file_sr = torchaudio.load(audio_file)
    
    # Resample if needed
    if file_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(file_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Extract segment if requested
    if duration is not None:
        start_sample = int(start_time * sample_rate)
        segment_samples = int(duration * sample_rate)
        end_sample = min(start_sample + segment_samples, waveform.shape[1])
        waveform = waveform[:, start_sample:end_sample]
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)
    
    return waveform

def benchmark_processor(processor, waveform, n_runs=5):
    """Benchmark a processor on a waveform."""
    # Run a warmup
    with torch.no_grad():
        _, _ = processor.process(waveform)
    
    # Run multiple trials
    process_times = []
    memory_kbs = []
    
    for _ in range(n_runs):
        # Measure processing time
        with torch.no_grad():
            start_time = time.time()
            output, metrics = processor.process(waveform)
            end_time = time.time()
        
        process_times.append(end_time - start_time)
        memory_kbs.append(metrics["memory_kb"])
    
    # Calculate averages
    avg_process_time = sum(process_times) / len(process_times)
    avg_memory_kb = sum(memory_kbs) / len(memory_kbs)
    
    # Create result summary
    result = {
        "avg_process_time": avg_process_time,
        "avg_memory_kb": avg_memory_kb,
        "tensor_shape": list(output.shape),
        "n_runs": n_runs
    }
    
    return result

def main():
    """Run benchmarks on processors."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple benchmark runner")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size (default: 1024)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length (default: 256)")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bands (default: 80)")
    parser.add_argument("--planes", type=str, default="default", 
                       help="Planes to include: 'default', 'all', 'minimal', or comma-separated list")
    parser.add_argument("--method", type=str, choices=["hps", "filterbank"], default="hps",
                        help="Harmonic plane method (default: hps)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive frequency resolution")
    parser.add_argument("--target_freq_bins", type=int, default=256,
                        help="Number of frequency bins with adaptive frequency (default: 256)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: entire file)")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs (default: 5)")
    
    args = parser.parse_args()
    
    # Parse planes configuration
    if args.planes == "default":
        include_planes = ['spectral', 'harmonic', 'spatial']
    elif args.planes == "all":
        include_planes = ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
    elif args.planes == "minimal":
        include_planes = ['spectral']
    else:
        include_planes = [p.strip() for p in args.planes.split(",")]
    
    # Load audio
    waveform = load_audio(
        args.audio_file, 
        sample_rate=args.sr, 
        duration=args.duration,
        start_time=args.start
    )
    
    print(f"Running benchmark on {args.audio_file}")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Running {args.n_runs} trials for each processor...\n")
    
    # Benchmark waveform processor
    print("Benchmarking waveform processor...")
    waveform_processor = WaveformProcessor(sample_rate=args.sr)
    waveform_result = benchmark_processor(waveform_processor, waveform, n_runs=args.n_runs)
    
    # Benchmark mel-spectrogram processor
    print("Benchmarking mel-spectrogram processor...")
    mel_processor = MelSpectrogramProcessor(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    mel_result = benchmark_processor(mel_processor, waveform, n_runs=args.n_runs)
    
    # Benchmark Wav2Tensor processor
    print("Benchmarking Wav2Tensor processor...")
    wav2tensor_processor = Wav2TensorProcessor(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        include_planes=include_planes,
        harmonic_method=args.method,
        use_adaptive_freq=args.adaptive,
        target_freq_bins=args.target_freq_bins
    )
    wav2tensor_result = benchmark_processor(wav2tensor_processor, waveform, n_runs=args.n_runs)
    
    # Print results
    print("\n==== BENCHMARK SUMMARY ====")
    print(f"Audio file: {args.audio_file}")
    
    if args.duration is not None:
        print(f"Segment: {args.start}s to {args.start + args.duration}s ({args.duration}s)")
    
    print("\nProcessing Time:")
    print(f"  Waveform:        {waveform_result['avg_process_time']:.6f} seconds")
    print(f"  Mel-spectrogram: {mel_result['avg_process_time']:.6f} seconds")
    print(f"  Wav2Tensor:      {wav2tensor_result['avg_process_time']:.6f} seconds")
    
    print("\nMemory Usage:")
    print(f"  Waveform:        {waveform_result['avg_memory_kb']:.2f} KB")
    print(f"  Mel-spectrogram: {mel_result['avg_memory_kb']:.2f} KB")
    print(f"  Wav2Tensor:      {wav2tensor_result['avg_memory_kb']:.2f} KB") 
    
    # Calculate speedup factors
    waveform_time = waveform_result['avg_process_time']
    mel_time = mel_result['avg_process_time']
    wav2tensor_time = wav2tensor_result['avg_process_time']
    
    speedup = {
        "wav2tensor_vs_waveform": waveform_time / wav2tensor_time if wav2tensor_time > 0 else float('inf'),
        "wav2tensor_vs_mel": mel_time / wav2tensor_time if wav2tensor_time > 0 else float('inf')
    }
    
    print("\nSpeedup Factors (higher is better):")
    print(f"  Wav2Tensor vs Waveform:         {speedup['wav2tensor_vs_waveform']:.2f}x")
    print(f"  Wav2Tensor vs Mel-spectrogram:  {speedup['wav2tensor_vs_mel']:.2f}x")
    
    print("\nOutput Tensor Shapes:")
    print(f"  Waveform:        {waveform_result['tensor_shape']}")
    print(f"  Mel-spectrogram: {mel_result['tensor_shape']}")
    print(f"  Wav2Tensor:      {wav2tensor_result['tensor_shape']}")
    
    # Save results to JSON file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    basename = os.path.splitext(os.path.basename(args.audio_file))[0]
    output_dir = "benchmarks/results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "audio_file": args.audio_file,
        "sample_rate": args.sr,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "include_planes": include_planes,
        "harmonic_method": args.method,
        "use_adaptive_freq": args.adaptive,
        "target_freq_bins": args.target_freq_bins,
        "timestamp": timestamp,
        "n_runs": args.n_runs,
        "waveform": waveform_result,
        "mel_spectrogram": mel_result,
        "wav2tensor": wav2tensor_result,
        "speedup": speedup,
        "output_dir": output_dir
    }
    
    output_file = f"{output_dir}/{basename}_{timestamp}_simple_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 