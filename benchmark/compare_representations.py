"""
Benchmark comparison for different audio representations.
This script compares computational costs (inference time and memory usage) between:
1. Raw waveform processing
2. Mel-spectrogram processing
3. Wav2Tensor (full version)
4. Wav2TensorLite
"""

import os
import time
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import argparse
import sys

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import processors
from benchmark.mel_spectrogram import MelSpectrogramProcessor
from benchmark.waveform import WaveformProcessor
from wav2tensor.core import Wav2TensorCore
from wav2tensor.core_lite import Wav2TensorLite

class BenchmarkSuite:
    """
    Benchmark suite for comparing different audio representations.
    """
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        include_planes: Optional[List[str]] = None,
        harmonic_method: str = 'hps',
        bit_depth: int = 16,
        fusion_method: str = 'concat',
        use_adaptive_freq: bool = False,
        target_freq_bins: int = 256,
        output_dir: str = './benchmark_results'
    ):
        """
        Initialize the benchmark suite.
        
        Args:
            sample_rate: Audio sample rate (default: 22050Hz)
            n_fft: FFT size for STFT (default: 1024)
            hop_length: Hop length for STFT (default: 256)
            n_mels: Number of Mel bands (default: 80)
            include_planes: List of planes to include for Wav2Tensor (default: None for all)
            harmonic_method: Harmonic plane method (default: 'hps')
            bit_depth: Bit depth for Wav2TensorLite (default: 16)
            fusion_method: Fusion method for Wav2TensorLite (default: 'concat')
            use_adaptive_freq: Whether to use adaptive frequency resolution for Wav2TensorLite (default: False)
            target_freq_bins: Number of frequency bins to use with adaptive frequency (default: 256)
            output_dir: Directory to save benchmark results (default: './benchmark_results')
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.include_planes = include_planes or ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        self.harmonic_method = harmonic_method
        self.bit_depth = bit_depth
        self.fusion_method = fusion_method
        self.use_adaptive_freq = use_adaptive_freq
        self.target_freq_bins = target_freq_bins
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize processors
        self.waveform_processor = WaveformProcessor(sample_rate=sample_rate)
        
        self.mel_processor = MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.wav2tensor = Wav2TensorCore(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins
        )
        
        self.wav2tensor_lite = Wav2TensorLite(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            bit_depth=bit_depth,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            fusion_method=fusion_method,
            use_adaptive_freq=use_adaptive_freq
        )
    
    def load_audio(self, audio_file: str, segment_duration: Optional[float] = None, start_time: float = 0.0) -> torch.Tensor:
        """
        Load audio file and extract segment if requested.
        
        Args:
            audio_file: Path to audio file
            segment_duration: Duration of segment to process in seconds (default: entire file)
            start_time: Start time in seconds (default: 0.0)
            
        Returns:
            waveform: Audio waveform tensor with shape [channels, samples]
        """
        # Load audio file
        waveform, file_sr = torchaudio.load(audio_file)
        
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
        
        return waveform
    
    def benchmark_waveform(self, waveform: torch.Tensor) -> Dict:
        """
        Benchmark waveform processing.
        
        Args:
            waveform: Audio waveform tensor with shape [channels, samples]
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        # Start timing
        start_time = time.time()
        
        # Process waveform
        processed_waveform, metrics = self.waveform_processor.process(waveform)
        
        # End timing
        process_time = time.time() - start_time
        
        # Override process time with more accurate measurement
        metrics["process_time"] = process_time
        
        # Add method identifier
        metrics["method"] = "waveform"
        
        return metrics
    
    def benchmark_mel_spectrogram(self, waveform: torch.Tensor) -> Dict:
        """
        Benchmark Mel-spectrogram processing.
        
        Args:
            waveform: Audio waveform tensor with shape [channels, samples]
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        # Start timing
        start_time = time.time()
        
        # Process waveform
        mel_spec, metrics = self.mel_processor.process(waveform)
        
        # End timing
        process_time = time.time() - start_time
        
        # Override process time with more accurate measurement
        metrics["process_time"] = process_time
        
        # Add method identifier
        metrics["method"] = "mel_spectrogram"
        
        return metrics
    
    def benchmark_wav2tensor(self, waveform: torch.Tensor) -> Dict:
        """
        Benchmark Wav2Tensor (full version) processing.
        
        Args:
            waveform: Audio waveform tensor with shape [channels, samples]
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        # Add batch dimension if necessary
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)
        
        # Start timing
        start_time = time.time()
        
        # Process with Wav2Tensor
        tensor, planes = self.wav2tensor(waveform)
        
        # End timing
        process_time = time.time() - start_time
        
        # Calculate memory usage
        memory_bytes = tensor.element_size() * tensor.nelement()
        memory_kb = memory_bytes / 1024
        
        # Collect metrics
        metrics = {
            "process_time": process_time,
            "memory_kb": memory_kb,
            "tensor_shape": list(tensor.shape),
            "method": "wav2tensor",
            "planes": {
                name: list(plane.shape) for name, plane in planes.items()
            }
        }
        
        return metrics
    
    def benchmark_wav2tensor_lite(self, waveform: torch.Tensor) -> Dict:
        """
        Benchmark Wav2TensorLite processing.
        
        Args:
            waveform: Audio waveform tensor with shape [channels, samples]
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        # Add batch dimension if necessary
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)
        
        # Start timing
        start_time = time.time()
        
        # Process with Wav2TensorLite
        tensor, metadata = self.wav2tensor_lite(waveform)
        
        # End timing
        process_time = time.time() - start_time
        
        # Override process time with more accurate measurement
        metadata["process_time"] = process_time
        
        # Add method identifier
        metadata["method"] = "wav2tensor_lite"
        
        # Calculate memory usage
        memory_bytes = tensor.element_size() * tensor.nelement()
        memory_kb = memory_bytes / 1024
        metadata["memory_kb"] = memory_kb
        
        return metadata
    
    def run_benchmark(self, audio_file: str, segment_duration: Optional[float] = None, start_time: float = 0.0, n_runs: int = 5) -> Dict:
        """
        Run benchmark for all methods and compare results.
        
        Args:
            audio_file: Path to audio file
            segment_duration: Duration of segment to process in seconds (default: entire file)
            start_time: Start time in seconds (default: 0.0)
            n_runs: Number of runs for each method (default: 5)
            
        Returns:
            results: Dictionary with benchmark results
        """
        # Load audio
        waveform = self.load_audio(audio_file, segment_duration, start_time)
        
        # Create results dictionary
        results = {
            "audio_file": audio_file,
            "segment_duration": segment_duration,
            "start_time": start_time,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "include_planes": self.include_planes,
            "harmonic_method": self.harmonic_method,
            "bit_depth": self.bit_depth,
            "fusion_method": self.fusion_method,
            "use_adaptive_freq": self.use_adaptive_freq,
            "waveform_shape": list(waveform.shape),
            "runs": []
        }
        
        # Run benchmark for each method multiple times
        for run_idx in range(n_runs):
            run_results = {
                "run_idx": run_idx,
                "waveform": self.benchmark_waveform(waveform),
                "mel_spectrogram": self.benchmark_mel_spectrogram(waveform),
                "wav2tensor": self.benchmark_wav2tensor(waveform),
                "wav2tensor_lite": self.benchmark_wav2tensor_lite(waveform)
            }
            results["runs"].append(run_results)
        
        # Calculate average metrics
        avg_results = {
            "waveform": {
                "avg_process_time": np.mean([run["waveform"]["process_time"] for run in results["runs"]]),
                "avg_memory_kb": np.mean([run["waveform"]["memory_kb"] for run in results["runs"]]),
                "tensor_shape": results["runs"][0]["waveform"]["tensor_shape"]
            },
            "mel_spectrogram": {
                "avg_process_time": np.mean([run["mel_spectrogram"]["process_time"] for run in results["runs"]]),
                "avg_memory_kb": np.mean([run["mel_spectrogram"]["memory_kb"] for run in results["runs"]]),
                "tensor_shape": results["runs"][0]["mel_spectrogram"]["tensor_shape"]
            },
            "wav2tensor": {
                "avg_process_time": np.mean([run["wav2tensor"]["process_time"] for run in results["runs"]]),
                "avg_memory_kb": np.mean([run["wav2tensor"]["memory_kb"] for run in results["runs"]]),
                "tensor_shape": results["runs"][0]["wav2tensor"]["tensor_shape"],
                "planes": results["runs"][0]["wav2tensor"]["planes"]
            },
            "wav2tensor_lite": {
                "avg_process_time": np.mean([run["wav2tensor_lite"]["process_time"] for run in results["runs"]]),
                "avg_memory_kb": np.mean([run["wav2tensor_lite"]["memory_kb"] for run in results["runs"]]),
                "tensor_shape": results["runs"][0]["wav2tensor_lite"]["tensor_shape"],
                "fusion_method": self.fusion_method,
                "bit_depth": self.bit_depth,
                "use_adaptive_freq": self.use_adaptive_freq
            }
        }
        
        # Add speed comparison
        avg_results["speedup"] = {
            "wav2tensor_vs_waveform": avg_results["waveform"]["avg_process_time"] / avg_results["wav2tensor"]["avg_process_time"],
            "wav2tensor_vs_mel": avg_results["mel_spectrogram"]["avg_process_time"] / avg_results["wav2tensor"]["avg_process_time"],
            "wav2tensor_lite_vs_wav2tensor": avg_results["wav2tensor"]["avg_process_time"] / avg_results["wav2tensor_lite"]["avg_process_time"],
            "wav2tensor_lite_vs_waveform": avg_results["waveform"]["avg_process_time"] / avg_results["wav2tensor_lite"]["avg_process_time"],
            "wav2tensor_lite_vs_mel": avg_results["mel_spectrogram"]["avg_process_time"] / avg_results["wav2tensor_lite"]["avg_process_time"]
        }
        
        # Add memory comparison
        avg_results["memory_reduction"] = {
            "wav2tensor_vs_waveform": avg_results["waveform"]["avg_memory_kb"] / avg_results["wav2tensor"]["avg_memory_kb"],
            "wav2tensor_vs_mel": avg_results["mel_spectrogram"]["avg_memory_kb"] / avg_results["wav2tensor"]["avg_memory_kb"],
            "wav2tensor_lite_vs_wav2tensor": avg_results["wav2tensor"]["avg_memory_kb"] / avg_results["wav2tensor_lite"]["avg_memory_kb"],
            "wav2tensor_lite_vs_waveform": avg_results["waveform"]["avg_memory_kb"] / avg_results["wav2tensor_lite"]["avg_memory_kb"],
            "wav2tensor_lite_vs_mel": avg_results["mel_spectrogram"]["avg_memory_kb"] / avg_results["wav2tensor_lite"]["avg_memory_kb"]
        }
        
        # Add average results to the main results dictionary
        results["avg_results"] = avg_results
        
        return results
    
    def _convert_for_json(self, obj):
        """
        Recursively convert objects to JSON-serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj

    def save_results(self, results: Dict, output_prefix: str = None):
        """
        Save benchmark results to file and generate plots.
        
        Args:
            results: Benchmark results dictionary
            output_prefix: Optional prefix for output files
        """
        # Create output prefix from audio file if not provided
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(results["audio_file"]))[0]
            segment_info = f"_{int(results.get('start_time', 0))}s_to_{int(results.get('start_time', 0) + results.get('segment_duration', 0))}s"
            output_prefix = f"{self.output_dir}/{base_name}{segment_info}"
        
        # Save results to JSON file
        json_path = f"{output_prefix}_benchmark.json"
        with open(json_path, 'w') as f:
            # Convert all objects to JSON-serializable types
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        print(f"Benchmark results saved to {json_path}")
        
        # Generate plots
        self._plot_process_time(results, output_prefix)
        self._plot_memory_usage(results, output_prefix)
        self._plot_speedup(results, output_prefix)
    
    def _plot_process_time(self, results: Dict, output_prefix: str):
        """Generate plot for processing time comparison."""
        avg_results = results["avg_results"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot processing time
        methods = ["waveform", "mel_spectrogram", "wav2tensor", "wav2tensor_lite"]
        times = [avg_results[method]["avg_process_time"] for method in methods]
        
        # Plot bars
        bars = plt.bar(methods, times)
        
        # Add labels
        plt.title("Processing Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.002,
                f"{height:.4f}s",
                ha='center', va='bottom', rotation=0
            )
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_process_time.png", dpi=300)
        plt.close()
        
        print(f"Processing time plot saved to {output_prefix}_process_time.png")
    
    def _plot_memory_usage(self, results: Dict, output_prefix: str):
        """Generate plot for memory usage comparison."""
        avg_results = results["avg_results"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot memory usage
        methods = ["waveform", "mel_spectrogram", "wav2tensor", "wav2tensor_lite"]
        memory = [avg_results[method]["avg_memory_kb"] for method in methods]
        
        # Plot bars
        bars = plt.bar(methods, memory)
        
        # Add labels
        plt.title("Memory Usage Comparison")
        plt.ylabel("Memory (KB)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.002,
                f"{height:.1f} KB",
                ha='center', va='bottom', rotation=0
            )
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_memory_usage.png", dpi=300)
        plt.close()
        
        print(f"Memory usage plot saved to {output_prefix}_memory_usage.png")
    
    def _plot_speedup(self, results: Dict, output_prefix: str):
        """Generate plot for speedup comparison."""
        avg_results = results["avg_results"]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot speedup
        comparisons = [
            "wav2tensor_vs_waveform",
            "wav2tensor_vs_mel",
            "wav2tensor_lite_vs_wav2tensor",
            "wav2tensor_lite_vs_waveform",
            "wav2tensor_lite_vs_mel"
        ]
        speedups = [avg_results["speedup"][comp] for comp in comparisons]
        
        # Format the labels for better readability
        labels = [comp.replace("_", " ").title() for comp in comparisons]
        
        # Plot bars
        bars = plt.bar(labels, speedups)
        
        # Add horizontal line at y=1 for reference
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        
        # Add labels
        plt.title("Speed Comparison (Speedup Factor)")
        plt.ylabel("Speedup Factor")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.05,
                f"{height:.2f}x",
                ha='center', va='bottom', rotation=0
            )
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_speedup.png", dpi=300)
        plt.close()
        
        print(f"Speedup plot saved to {output_prefix}_speedup.png")

    def print_summary(self, results: Dict):
        """
        Print summary of benchmark results.
        
        Args:
            results: Benchmark results dictionary
        """
        avg_results = results["avg_results"]
        
        print("\n==== BENCHMARK SUMMARY ====")
        print(f"Audio file: {results['audio_file']}")
        
        if results.get('segment_duration') is not None:
            print(f"Segment: {results['start_time']}s to {results['start_time'] + results['segment_duration']}s ({results['segment_duration']}s)")
        
        print("\nProcessing Time:")
        print(f"  Waveform:        {avg_results['waveform']['avg_process_time']:.4f} seconds")
        print(f"  Mel-spectrogram: {avg_results['mel_spectrogram']['avg_process_time']:.4f} seconds")
        print(f"  Wav2Tensor:      {avg_results['wav2tensor']['avg_process_time']:.4f} seconds")
        print(f"  Wav2TensorLite:  {avg_results['wav2tensor_lite']['avg_process_time']:.4f} seconds")
        
        print("\nMemory Usage:")
        print(f"  Waveform:        {avg_results['waveform']['avg_memory_kb']:.2f} KB")
        print(f"  Mel-spectrogram: {avg_results['mel_spectrogram']['avg_memory_kb']:.2f} KB")
        print(f"  Wav2Tensor:      {avg_results['wav2tensor']['avg_memory_kb']:.2f} KB")
        print(f"  Wav2TensorLite:  {avg_results['wav2tensor_lite']['avg_memory_kb']:.2f} KB")
        
        print("\nSpeedup Factors:")
        print(f"  Wav2Tensor vs Waveform:          {avg_results['speedup']['wav2tensor_vs_waveform']:.2f}x")
        print(f"  Wav2Tensor vs Mel-spectrogram:   {avg_results['speedup']['wav2tensor_vs_mel']:.2f}x")
        print(f"  Wav2TensorLite vs Wav2Tensor:    {avg_results['speedup']['wav2tensor_lite_vs_wav2tensor']:.2f}x")
        print(f"  Wav2TensorLite vs Waveform:      {avg_results['speedup']['wav2tensor_lite_vs_waveform']:.2f}x")
        print(f"  Wav2TensorLite vs Mel-spectrogram: {avg_results['speedup']['wav2tensor_lite_vs_mel']:.2f}x")
        
        print("\nMemory Reduction Factors:")
        print(f"  Wav2Tensor vs Waveform:          {avg_results['memory_reduction']['wav2tensor_vs_waveform']:.2f}x")
        print(f"  Wav2Tensor vs Mel-spectrogram:   {avg_results['memory_reduction']['wav2tensor_vs_mel']:.2f}x")
        print(f"  Wav2TensorLite vs Wav2Tensor:    {avg_results['memory_reduction']['wav2tensor_lite_vs_wav2tensor']:.2f}x")
        print(f"  Wav2TensorLite vs Waveform:      {avg_results['memory_reduction']['wav2tensor_lite_vs_waveform']:.2f}x")
        print(f"  Wav2TensorLite vs Mel-spectrogram: {avg_results['memory_reduction']['wav2tensor_lite_vs_mel']:.2f}x")
        
        print("\nTensor Shapes:")
        print(f"  Waveform:        {avg_results['waveform']['tensor_shape']}")
        print(f"  Mel-spectrogram: {avg_results['mel_spectrogram']['tensor_shape']}")
        print(f"  Wav2Tensor:      {avg_results['wav2tensor']['tensor_shape']}")
        print(f"  Wav2TensorLite:  {avg_results['wav2tensor_lite']['tensor_shape']}")
        
        print("\nWav2TensorLite Configuration:")
        print(f"  Fusion Method:   {avg_results['wav2tensor_lite']['fusion_method']}")
        print(f"  Bit Depth:       {avg_results['wav2tensor_lite']['bit_depth']}")
        print(f"  Adaptive Freq:   {avg_results['wav2tensor_lite']['use_adaptive_freq']}")
        
        print("\n==== END OF SUMMARY ====\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark comparison for different audio representations")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (default: 3.0)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size (default: 1024)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length (default: 256)")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bands (default: 80)")
    parser.add_argument("--planes", type=str, default="default", 
                        help="Planes to include, comma-separated (default: 'default' for spectral,harmonic,spatial)")
    parser.add_argument("--method", type=str, default="hps", choices=["hps", "filterbank"], 
                        help="Harmonic plane method (default: hps)")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[8, 16], 
                        help="Bit depth for Wav2TensorLite (default: 16)")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "add", "learned"], 
                        help="Fusion method for Wav2TensorLite (default: concat)")
    parser.add_argument("--adaptive", action="store_true", 
                        help="Use adaptive frequency resolution for Wav2TensorLite")
    parser.add_argument("--target-freq-bins", type=int, default=256,
                        help="Number of frequency bins to use with adaptive frequency (default: 256)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", 
                        help="Output directory (default: benchmark_results)")
    parser.add_argument("--n-runs", type=int, default=5, 
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