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
        
        # Calculate speedup factors
        waveform_time = waveform_metrics['avg_process_time']
        mel_time = mel_metrics['avg_process_time']
        wav2tensor_time = wav2tensor_metrics['avg_process_time']
        
        speedup = {
            "wav2tensor_vs_waveform": waveform_time / wav2tensor_time if wav2tensor_time > 0 else float('inf'),
            "wav2tensor_vs_mel": mel_time / wav2tensor_time if wav2tensor_time > 0 else float('inf')
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
            "use_adaptive_freq": self.use_adaptive_freq,
            "target_freq_bins": self.target_freq_bins,
            "waveform_shape": list(waveform.shape),
            "waveform": waveform_metrics,
            "mel_spectrogram": mel_metrics,
            "wav2tensor": wav2tensor_metrics,
            "speedup": speedup,
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "output_dir": self.output_dir
        }
        
        return results
    
    def save_results(self, results: Dict, output_prefix: Optional[str] = None):
        """
        Save benchmark results to file.
        
        Args:
            results: Dictionary with benchmark results
            output_prefix: Optional prefix for output file name
        """
        # Create output file name
        timestamp = results.get("timestamp", time.strftime("%Y%m%d-%H%M%S"))
        basename = os.path.splitext(os.path.basename(results["audio_file"]))[0]
        
        if output_prefix:
            output_file = f"{self.output_dir}/{output_prefix}_{timestamp}.json"
        else:
            output_file = f"{self.output_dir}/{basename}_{timestamp}_benchmark.json"
        
        # Convert numpy/torch/etc. types to Python native types
        def convert_for_json(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist() if obj.numel() > 0 else []
            elif isinstance(obj, Dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, List):
                return [convert_for_json(i) for i in obj]
            return obj
        
        # Convert data and save to file
        converted_results = convert_for_json(results)
        with open(output_file, "w") as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """
        Print a summary of benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        print("\n==== BENCHMARK SUMMARY ====")
        print(f"Audio file: {results['audio_file']}")
        
        if results["segment_duration"] is not None:
            print(f"Segment: {results['start_time']}s to {results['start_time'] + results['segment_duration']}s ({results['segment_duration']}s)")
        
        print("\nProcessing Time:")
        print(f"  Waveform:        {results['waveform']['avg_process_time']:.6f} seconds")
        print(f"  Mel-spectrogram: {results['mel_spectrogram']['avg_process_time']:.6f} seconds")
        print(f"  Wav2Tensor:      {results['wav2tensor']['avg_process_time']:.6f} seconds")
        
        print("\nMemory Usage:")
        print(f"  Waveform:        {results['waveform']['avg_memory_kb']:.2f} KB")
        print(f"  Mel-spectrogram: {results['mel_spectrogram']['avg_memory_kb']:.2f} KB")
        print(f"  Wav2Tensor:      {results['wav2tensor']['avg_memory_kb']:.2f} KB")
        
        print("\nSpeedup Factors (higher is better):")
        print(f"  Wav2Tensor vs Waveform:         {results['speedup']['wav2tensor_vs_waveform']:.2f}x")
        print(f"  Wav2Tensor vs Mel-spectrogram:  {results['speedup']['wav2tensor_vs_mel']:.2f}x")
        
        print("\nOutput Tensor Shapes:")
        # Use tensor_shape directly from the results
        print(f"  Waveform:        {results['waveform'].get('tensor_shape', 'N/A')}")
        print(f"  Mel-spectrogram: {results['mel_spectrogram'].get('tensor_shape', 'N/A')}")
        print(f"  Wav2Tensor:      {results['wav2tensor'].get('tensor_shape', 'N/A')}")


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