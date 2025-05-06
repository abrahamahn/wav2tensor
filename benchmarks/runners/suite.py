"""
BenchmarkSuite for comparing different audio representations.

This module provides a BenchmarkSuite class that benchmarks and compares
different audio representation methods.
"""

import os
import time
import json
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Optional, Tuple, Any

from benchmarks.processors.waveform import WaveformProcessor
from benchmarks.processors.mel_spectrogram import MelSpectrogramProcessor
from benchmarks.processors.wav2tensor import Wav2TensorProcessor
from benchmarks.processors.wav2tensor_lite import Wav2TensorLiteProcessor


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
        output_dir: str = 'benchmarks/results'
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
            use_adaptive_freq: Whether to use adaptive frequency resolution (default: False)
            target_freq_bins: Number of frequency bins to use with adaptive frequency (default: 256)
            output_dir: Directory to save benchmark results (default: 'benchmarks/results')
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
        
        self.wav2tensor_processor = Wav2TensorProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins
        )
        
        self.wav2tensor_lite_processor = Wav2TensorLiteProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            bit_depth=bit_depth,
            fusion_method=fusion_method,
            use_adaptive_freq=use_adaptive_freq
        )
    
    def load_audio(self, audio_file: str, segment_duration: Optional[float] = None, start_time: float = 0.0) -> torch.Tensor:
        """
        Load audio file and prepare for benchmarking.
        
        Args:
            audio_file: Path to audio file
            segment_duration: Duration of segment to process in seconds (default: entire file)
            start_time: Start time in seconds (default: 0.0)
            
        Returns:
            waveform: Audio waveform tensor with shape [batch, channels, samples]
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
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0)
        
        return waveform
    
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
        
        print(f"Running benchmark on {audio_file}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Running {n_runs} trials for each processor...")
        
        # Benchmark waveform processor
        print("Benchmarking waveform processor...")
        waveform_metrics = self.waveform_processor.benchmark(waveform, n_runs=n_runs)
        
        # Benchmark mel-spectrogram processor
        print("Benchmarking mel-spectrogram processor...")
        mel_metrics = self.mel_processor.benchmark(waveform, n_runs=n_runs)
        
        # Benchmark Wav2Tensor processor
        print("Benchmarking Wav2Tensor processor...")
        wav2tensor_metrics = self.wav2tensor_processor.benchmark(waveform, n_runs=n_runs)
        
        # Benchmark Wav2TensorLite processor
        print("Benchmarking Wav2TensorLite processor...")
        wav2tensor_lite_metrics = self.wav2tensor_lite_processor.benchmark(waveform, n_runs=n_runs)
        
        # Calculate speedup factors
        waveform_time = waveform_metrics['avg_process_time']
        mel_time = mel_metrics['avg_process_time']
        wav2tensor_time = wav2tensor_metrics['avg_process_time']
        wav2tensor_lite_time = wav2tensor_lite_metrics['avg_process_time']
        
        speedup = {
            "wav2tensor_vs_waveform": waveform_time / wav2tensor_time if wav2tensor_time > 0 else float('inf'),
            "wav2tensor_vs_mel": mel_time / wav2tensor_time if wav2tensor_time > 0 else float('inf'),
            "wav2tensor_lite_vs_wav2tensor": wav2tensor_time / wav2tensor_lite_time if wav2tensor_lite_time > 0 else float('inf'),
            "wav2tensor_lite_vs_waveform": waveform_time / wav2tensor_lite_time if wav2tensor_lite_time > 0 else float('inf'),
            "wav2tensor_lite_vs_mel": mel_time / wav2tensor_lite_time if wav2tensor_lite_time > 0 else float('inf')
        }
        
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
            "waveform": waveform_metrics,
            "mel_spectrogram": mel_metrics,
            "wav2tensor": wav2tensor_metrics,
            "wav2tensor_lite": wav2tensor_lite_metrics,
            "speedup": speedup
        }
        
        return results
    
    def save_results(self, results: Dict, output_prefix: Optional[str] = None):
        """
        Save benchmark results to file.
        
        Args:
            results: Benchmark results dictionary
            output_prefix: Optional prefix for output files
        """
        # Create output prefix from audio file if not provided
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(results["audio_file"]))[0]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_prefix = f"{self.output_dir}/{base_name}_{timestamp}"
        
        # Save results to JSON file
        json_path = f"{output_prefix}_benchmark.json"
        
        # Convert tensors and numpy arrays to lists
        def convert_for_json(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (bool, int, float, str)) or obj is None:
                return obj
            else:
                return str(obj)
        
        json_results = convert_for_json(results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Benchmark results saved to {json_path}")
    
    def print_summary(self, results: Dict):
        """
        Print summary of benchmark results.
        
        Args:
            results: Benchmark results dictionary
        """
        print("\n==== BENCHMARK SUMMARY ====")
        print(f"Audio file: {results['audio_file']}")
        
        if results.get('segment_duration') is not None:
            print(f"Segment: {results['start_time']}s to {results['start_time'] + results['segment_duration']}s ({results['segment_duration']}s)")
        
        print("\nProcessing Time:")
        print(f"  Waveform:        {results['waveform']['avg_process_time']:.6f} seconds")
        print(f"  Mel-spectrogram: {results['mel_spectrogram']['avg_process_time']:.6f} seconds")
        print(f"  Wav2Tensor:      {results['wav2tensor']['avg_process_time']:.6f} seconds")
        print(f"  Wav2TensorLite:  {results['wav2tensor_lite']['avg_process_time']:.6f} seconds")
        
        print("\nMemory Usage:")
        print(f"  Waveform:        {results['waveform']['avg_memory_kb']:.2f} KB")
        print(f"  Mel-spectrogram: {results['mel_spectrogram']['avg_memory_kb']:.2f} KB")
        print(f"  Wav2Tensor:      {results['wav2tensor']['avg_memory_kb']:.2f} KB")
        print(f"  Wav2TensorLite:  {results['wav2tensor_lite']['avg_memory_kb']:.2f} KB")
        
        print("\nSpeedup Factors (higher is better):")
        print(f"  Wav2Tensor vs Waveform:          {results['speedup']['wav2tensor_vs_waveform']:.2f}x")
        print(f"  Wav2Tensor vs Mel-spectrogram:   {results['speedup']['wav2tensor_vs_mel']:.2f}x")
        print(f"  Wav2TensorLite vs Wav2Tensor:    {results['speedup']['wav2tensor_lite_vs_wav2tensor']:.2f}x")
        print(f"  Wav2TensorLite vs Waveform:      {results['speedup']['wav2tensor_lite_vs_waveform']:.2f}x")
        print(f"  Wav2TensorLite vs Mel-spectrogram: {results['speedup']['wav2tensor_lite_vs_mel']:.2f}x")
        
        print("\nOutput Tensor Shapes:")
        print(f"  Waveform:        {results['waveform']['tensor_shape']}")
        print(f"  Mel-spectrogram: {results['mel_spectrogram']['tensor_shape']}")
        print(f"  Wav2Tensor:      {results['wav2tensor']['tensor_shape']}")
        print(f"  Wav2TensorLite:  {results['wav2tensor_lite']['tensor_shape']}")


if __name__ == "__main__":
    import argparse
    
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