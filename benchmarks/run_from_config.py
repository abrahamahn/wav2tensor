#!/usr/bin/env python3
"""
Config-based benchmark runner for Wav2Tensor.

Run benchmarks using configuration from a YAML file.
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
from benchmarks.configs.config_loader import load_config


def main():
    """Run benchmark suite using configuration from a YAML file."""
    parser = argparse.ArgumentParser(description="Run benchmark suite using a configuration file")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output_dir", type=str, help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration file
    config = load_config(args.config_file)
    
    # Override output directory if specified
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        include_planes=config["include_planes"],
        harmonic_method=config["harmonic_method"],
        use_adaptive_freq=config["use_adaptive_freq"],
        target_freq_bins=config["target_freq_bins"],
        output_dir=config["output_dir"]
    )
    
    # Run benchmark
    results = benchmark_suite.run_benchmark(
        audio_file=args.audio_file,
        segment_duration=config["segment_duration"],
        start_time=config["start_time"],
        n_runs=config["n_runs"]
    )
    
    # Print summary
    benchmark_suite.print_summary(results)
    
    # Save results
    benchmark_suite.save_results(results)


if __name__ == "__main__":
    main() 