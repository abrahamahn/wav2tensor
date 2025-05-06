#!/usr/bin/env python3
"""
Benchmark runner for Wav2Tensor.

Run benchmarks to compare different audio representation methods.
"""

import os
import sys
import argparse

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply monkey patches before importing other modules
from benchmarks.monkey_patch import apply_patches
apply_patches()

from benchmarks.runners.suite import BenchmarkSuite


def main():
    """Run benchmark suite on an audio file."""
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size (default: 1024)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length (default: 256)")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bands (default: 80)")
    parser.add_argument("--planes", type=str, default="default",
                        help="Planes to include: 'default', 'all', 'minimal', or comma-separated list")
    parser.add_argument("--method", type=str, choices=["hps", "filterbank"], default="hps",
                        help="Harmonic plane method (default: hps)")
    parser.add_argument("--bit_depth", type=int, choices=[8, 16], default=16,
                        help="Bit depth for Wav2TensorLite quantization (default: 16)")
    parser.add_argument("--fusion", type=str, choices=["concat", "add", "learned"], default="concat",
                        help="Fusion method for Wav2TensorLite (default: concat)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive frequency resolution")
    parser.add_argument("--target_freq_bins", type=int, default=256,
                        help="Number of frequency bins with adaptive frequency (default: 256)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: entire file)")
    parser.add_argument("--output_dir", type=str, default="benchmarks/results",
                        help="Directory to save benchmark results (default: 'benchmarks/results')")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs for each method (default: 5)")
    
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
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        include_planes=include_planes,
        harmonic_method=args.method,
        bit_depth=args.bit_depth,
        fusion_method=args.fusion,
        use_adaptive_freq=args.adaptive,
        target_freq_bins=args.target_freq_bins,
        output_dir=args.output_dir
    )
    
    # Run benchmark
    results = benchmark_suite.run_benchmark(
        audio_file=args.audio_file,
        segment_duration=args.duration,
        start_time=args.start,
        n_runs=args.n_runs
    )
    
    # Print summary
    benchmark_suite.print_summary(results)
    
    # Save results
    benchmark_suite.save_results(results)


if __name__ == "__main__":
    main() 