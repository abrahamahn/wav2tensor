"""
BenchmarkSuite for comparing different audio representations.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch
import torchaudio

from benchmarks.processors.mel_spectrogram import MelSpectrogramProcessor
from benchmarks.processors.wav2tensor import Wav2TensorProcessor
from benchmarks.processors.waveform import WaveformProcessor
from wav2tensor.frequency import VALID_PLANES


class BenchmarkSuite:
    """Benchmark suite for comparing different audio representations."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        include_planes: list[str] | None = None,
        harmonic_method: str = "hps",
        use_adaptive_freq: bool = False,
        target_freq_bins: int = 256,
        output_dir: str = "benchmarks/results",
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.include_planes = include_planes or list(VALID_PLANES)
        self.harmonic_method = harmonic_method
        self.use_adaptive_freq = use_adaptive_freq
        self.target_freq_bins = target_freq_bins
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.waveform_processor = WaveformProcessor(sample_rate=sample_rate)
        self.mel_processor = MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.wav2tensor_processor = Wav2TensorProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=self.include_planes,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins,
        )

    def load_audio(
        self,
        audio_file: str,
        segment_duration: float | None = None,
        start_time: float = 0.0,
    ) -> torch.Tensor:
        """Load audio file and prepare [batch, channels, samples] waveform tensor."""
        waveform, file_sr = torchaudio.load(audio_file)
        if file_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(file_sr, self.sample_rate)
            waveform = resampler(waveform)

        if segment_duration is not None:
            start_sample = int(start_time * self.sample_rate)
            segment_samples = int(segment_duration * self.sample_rate)
            end_sample = min(start_sample + segment_samples, waveform.shape[1])
            waveform = waveform[:, start_sample:end_sample]

        return waveform.unsqueeze(0)

    def run_benchmark(
        self,
        audio_file: str,
        segment_duration: float | None = None,
        start_time: float = 0.0,
        n_runs: int = 5,
    ) -> dict[str, Any]:
        """Run benchmark for all methods and compare results."""
        waveform = self.load_audio(audio_file, segment_duration, start_time)

        waveform_metrics = self.waveform_processor.benchmark(waveform, n_runs=n_runs)
        mel_metrics = self.mel_processor.benchmark(waveform, n_runs=n_runs)
        wav2tensor_metrics = self.wav2tensor_processor.benchmark(waveform, n_runs=n_runs)

        waveform_time = waveform_metrics["avg_process_time"]
        mel_time = mel_metrics["avg_process_time"]
        wav2tensor_time = wav2tensor_metrics["avg_process_time"]

        speedup = {
            "wav2tensor_vs_waveform": (waveform_time / wav2tensor_time if wav2tensor_time > 0 else float("inf")),
            "wav2tensor_vs_mel": (mel_time / wav2tensor_time if wav2tensor_time > 0 else float("inf")),
        }

        return {
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
            "output_dir": self.output_dir,
        }

    def save_results(self, results: dict[str, Any], output_prefix: str | None = None) -> None:
        """Save benchmark results to JSON file."""
        timestamp = results.get("timestamp", time.strftime("%Y%m%d-%H%M%S"))
        basename = os.path.splitext(os.path.basename(results["audio_file"]))[0]

        if output_prefix:
            output_file = f"{self.output_dir}/{output_prefix}_{timestamp}.json"
        else:
            output_file = f"{self.output_dir}/{basename}_{timestamp}_benchmark.json"

        def convert_for_json(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist() if obj.numel() > 0 else []
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            return obj

        converted_results = convert_for_json(results)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted_results, f, indent=2)

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print a summary of benchmark results."""
        print("\n==== BENCHMARK SUMMARY ====")
        print(f"Audio file: {results['audio_file']}")
        if results["segment_duration"] is not None:
            end_sec = results["start_time"] + results["segment_duration"]
            print(f"Segment: {results['start_time']}s to {end_sec}s ({results['segment_duration']}s)")

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
        print(f"  Waveform:        {results['waveform'].get('tensor_shape', 'N/A')}")
        print(f"  Mel-spectrogram: {results['mel_spectrogram'].get('tensor_shape', 'N/A')}")
        print(f"  Wav2Tensor:      {results['wav2tensor'].get('tensor_shape', 'N/A')}")
