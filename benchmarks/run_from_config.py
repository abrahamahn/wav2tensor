#!/usr/bin/env python3
"""
Config-based benchmark runner for Wav2Tensor.

Run benchmarks using configuration from a YAML file.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.configs.config_loader import load_config
from benchmarks.runners.suite import BenchmarkSuite


def main() -> None:
    """Run benchmark suite using configuration from a YAML file."""
    parser = argparse.ArgumentParser(description="Run benchmark suite using a configuration file")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output_dir", type=str, help="Override output directory from config")

    args = parser.parse_args()
    config = load_config(args.config_file)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    os.makedirs(config["output_dir"], exist_ok=True)

    benchmark_suite = BenchmarkSuite(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        include_planes=config["include_planes"],
        harmonic_method=config["harmonic_method"],
        use_adaptive_freq=config["use_adaptive_freq"],
        target_freq_bins=config["target_freq_bins"],
        output_dir=config["output_dir"],
    )

    results = benchmark_suite.run_benchmark(
        audio_file=args.audio_file,
        segment_duration=config["segment_duration"],
        start_time=config["start_time"],
        n_runs=config["n_runs"],
    )

    benchmark_suite.print_summary(results)
    benchmark_suite.save_results(results)


if __name__ == "__main__":
    main()
