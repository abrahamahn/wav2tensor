"""
Wav2Tensor processor for benchmarking.

This module implements a processor for Wav2Tensor representation with benchmarking capabilities.
"""

import torch
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wav2tensor.core import Wav2TensorCore
from benchmarks.processors.base import BaseProcessor


class Wav2TensorProcessor(BaseProcessor):
    """Processor for Wav2Tensor representation with performance tracking."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        include_planes: Optional[List[str]] = None,
        harmonic_method: str = 'hps',
        use_adaptive_freq: bool = False,
        target_freq_bins: int = 256
    ):
        """
        Initialize the Wav2Tensor processor.
        
        Args:
            sample_rate: Audio sample rate (default: 22050Hz)
            n_fft: FFT size for STFT (default: 1024)
            hop_length: Hop length for STFT (default: 256)
            include_planes: List of planes to include (default: None for all)
            harmonic_method: Harmonic plane method (default: 'hps')
            use_adaptive_freq: Whether to use adaptive frequency resolution (default: False)
            target_freq_bins: Number of frequency bins with adaptive frequency (default: 256)
        """
        super().__init__(sample_rate=sample_rate, name="wav2tensor")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_planes = include_planes or ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
        self.harmonic_method = harmonic_method
        self.use_adaptive_freq = use_adaptive_freq
        self.target_freq_bins = target_freq_bins
        
        # Initialize Wav2Tensor with jit_script=False to avoid the _ensure_4d issue
        self.wav2tensor = Wav2TensorCore(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins
        )
        
        # Define our own ensure_4d function to avoid using the one with @torch.jit.script
        self.ensure_4d = lambda tensor: tensor.unsqueeze(1) if len(tensor.shape) == 3 else tensor
    
    def process(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process an audio waveform into a Wav2Tensor representation and measure performance.
        
        Args:
            waveform: Audio waveform tensor with shape [batch, channels, samples]
            
        Returns:
            tensor: Wav2Tensor representation tensor
            metrics: Dictionary with performance metrics
        """
        # Process the waveform using the Wav2TensorCore instance
        try:
            with torch.no_grad():  # Use no_grad for benchmarking
                tensor, planes = self.wav2tensor(waveform)
            
            # Calculate memory usage
            memory_bytes = tensor.element_size() * tensor.nelement()
            memory_kb = memory_bytes / 1024
            
            # Record plane information
            planes_info = {}
            for plane_name, plane_tensor in planes.items():
                planes_info[plane_name] = list(plane_tensor.shape)
            
            # Collect metrics
            metrics = {
                "memory_kb": memory_kb,
                "tensor_shape": list(tensor.shape),
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "harmonic_method": self.harmonic_method,
                "use_adaptive_freq": self.use_adaptive_freq,
                "target_freq_bins": self.target_freq_bins,
                "planes": planes_info
            }
            
            return tensor, metrics
            
        except Exception as e:
            # If there's an error, process each plane individually to avoid dimension issues
            with torch.no_grad():
                # Process spectral plane only first as a fallback
                original_include_planes = self.wav2tensor.include_planes
                self.wav2tensor.include_planes = ['spectral']
                tensor, planes = self.wav2tensor(waveform)
                
                # Process other planes individually if needed
                planes_info = {}
                for plane_name, plane_tensor in planes.items():
                    planes_info[plane_name] = list(plane_tensor.shape)
                
                # Reset include_planes
                self.wav2tensor.include_planes = original_include_planes
                
                # Calculate memory usage for the spectral plane
                memory_bytes = tensor.element_size() * tensor.nelement()
                memory_kb = memory_bytes / 1024
                
                # Collect limited metrics
                metrics = {
                    "memory_kb": memory_kb,
                    "tensor_shape": list(tensor.shape),
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "harmonic_method": self.harmonic_method,
                    "use_adaptive_freq": self.use_adaptive_freq,
                    "target_freq_bins": self.target_freq_bins,
                    "planes": planes_info,
                    "error": str(e),
                    "note": "Error occurred, processed spectral plane only"
                }
                
                return tensor, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process waveform with Wav2Tensor and benchmark")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size (default: 1024)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length (default: 256)")
    parser.add_argument("--planes", type=str, default="spectral,harmonic,spatial,psychoacoustic", 
                       help="Comma-separated list of planes to include")
    parser.add_argument("--harmonic_method", type=str, choices=["hps", "filterbank"], default="hps",
                       help="Harmonic plane method (default: hps)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive frequency resolution")
    parser.add_argument("--target_freq_bins", type=int, default=256, 
                       help="Number of frequency bins with adaptive frequency")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (default: entire file)")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    
    args = parser.parse_args()
    
    # Parse planes list
    planes = args.planes.split(",")
    
    # Initialize processor
    processor = Wav2TensorProcessor(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        include_planes=planes,
        harmonic_method=args.harmonic_method,
        use_adaptive_freq=args.adaptive,
        target_freq_bins=args.target_freq_bins
    )
    
    # Process audio file
    output, metrics = processor.process_file(
        audio_file=args.audio_file,
        segment_duration=args.duration,
        start_time=args.start
    )
    
    # Benchmark
    import torchaudio
    waveform, _ = torchaudio.load(args.audio_file)
    
    # Extract segment if requested
    if args.duration is not None:
        start_sample = int(args.start * args.sr)
        segment_samples = int(args.duration * args.sr)
        end_sample = min(start_sample + segment_samples, waveform.shape[1])
        waveform = waveform[:, start_sample:end_sample]
    
    # Benchmark
    metrics = processor.benchmark(waveform, n_runs=args.n_runs)
    
    # Print benchmark results
    print(f"Wav2Tensor processor benchmark results ({args.n_runs} runs):")
    print(f"  Average processing time: {metrics['avg_process_time']:.6f} seconds")
    print(f"  Memory usage: {metrics['avg_memory_kb']:.2f} KB")
    print(f"  Shape: {metrics['tensor_shape']}")
    print(f"  Planes:")
    for plane_name, plane_shape in metrics["planes"].items():
        print(f"    {plane_name}: {plane_shape}") 