"""
Generate test audio files for reconstruction quality testing.

This script creates synthetic audio files with various characteristics
to test the reconstruction quality of different audio representations.
"""

import os
import torch
import torchaudio
import numpy as np
import argparse

def generate_sine_sweep(duration=5.0, sample_rate=22050, min_freq=20, max_freq=20000):
    """Generate a logarithmic sine sweep from min_freq to max_freq"""
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    # Logarithmic sweep formula
    log_ratio = np.log(max_freq / min_freq)
    phase = 2 * np.pi * min_freq * duration / log_ratio * (torch.exp(t / duration * log_ratio) - 1)
    sweep = torch.sin(phase)
    
    # Normalize
    sweep = sweep / torch.max(torch.abs(sweep))
    
    # Return as [1, num_samples] tensor
    return sweep.unsqueeze(0)

def generate_harmonic_tones(duration=5.0, sample_rate=22050, fundamental=220.0):
    """Generate a signal with harmonic overtones at integer multiples of fundamental frequency"""
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    # Create fundamental
    signal = torch.sin(2 * np.pi * fundamental * t)
    
    # Add harmonics with decreasing amplitude
    for i in range(2, 11):  # Add 9 harmonics
        harmonic_freq = fundamental * i
        if harmonic_freq < sample_rate / 2:  # Stay below Nyquist
            harmonic = torch.sin(2 * np.pi * harmonic_freq * t) / i
            signal += harmonic
    
    # Normalize
    signal = signal / torch.max(torch.abs(signal))
    
    # Return as [1, num_samples] tensor
    return signal.unsqueeze(0)

def generate_noise_burst(duration=5.0, sample_rate=22050, burst_duration=0.1, interval=1.0):
    """Generate periodic noise bursts with silence between"""
    num_samples = int(duration * sample_rate)
    signal = torch.zeros(num_samples)
    
    burst_samples = int(burst_duration * sample_rate)
    interval_samples = int(interval * sample_rate)
    
    # Add noise bursts at regular intervals
    for start in range(0, num_samples, interval_samples):
        end = min(start + burst_samples, num_samples)
        signal[start:end] = torch.randn(end - start)
    
    # Normalize
    signal = signal / torch.max(torch.abs(signal))
    
    # Return as [1, num_samples] tensor
    return signal.unsqueeze(0)

def generate_stereo_panning(duration=5.0, sample_rate=22050, pan_freq=0.2):
    """Generate a tone that pans between left and right channels"""
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    # Create a 440Hz tone
    mono = torch.sin(2 * np.pi * 440 * t)
    
    # Create panning envelope (0 to 1)
    pan = 0.5 + 0.5 * torch.sin(2 * np.pi * pan_freq * t)
    
    # Create stereo signal with panning
    left = mono * (1 - pan)
    right = mono * pan
    
    # Stack into stereo signal
    stereo = torch.stack([left, right])
    
    # Normalize
    stereo = stereo / torch.max(torch.abs(stereo))
    
    return stereo

def generate_complex_mix(duration=5.0, sample_rate=22050):
    """Generate a complex mix of tones, noise, and transients"""
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    # Base tone at 110Hz with harmonics
    signal = torch.sin(2 * np.pi * 110 * t)
    signal += 0.5 * torch.sin(2 * np.pi * 220 * t)
    signal += 0.25 * torch.sin(2 * np.pi * 330 * t)
    signal += 0.125 * torch.sin(2 * np.pi * 440 * t)
    
    # Add a higher frequency tone that starts later
    high_tone = torch.sin(2 * np.pi * 880 * t)
    mask = torch.zeros_like(t)
    mask[int(num_samples * 0.4):] = 1
    signal += 0.3 * high_tone * mask
    
    # Add some noise bursts
    for start_time in [1.0, 2.0, 3.0, 4.0]:
        start_idx = int(start_time * sample_rate)
        end_idx = min(start_idx + int(0.05 * sample_rate), num_samples)
        signal[start_idx:end_idx] += 0.5 * torch.randn(end_idx - start_idx)
    
    # Add some background noise
    signal += 0.05 * torch.randn(num_samples)
    
    # Normalize
    signal = signal / torch.max(torch.abs(signal))
    
    # Return as [1, num_samples] tensor
    return signal.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Generate test audio files for reconstruction quality testing")
    parser.add_argument("--output", type=str, default="test_audio", help="Output directory")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate and save test files
    print("Generating test audio files...")
    
    # Sine sweep
    sweep = generate_sine_sweep(args.duration, args.sr)
    torchaudio.save(os.path.join(args.output, "sine_sweep.wav"), sweep, args.sr)
    
    # Harmonic tones
    tones = generate_harmonic_tones(args.duration, args.sr)
    torchaudio.save(os.path.join(args.output, "harmonic_tones.wav"), tones, args.sr)
    
    # Noise bursts
    bursts = generate_noise_burst(args.duration, args.sr)
    torchaudio.save(os.path.join(args.output, "noise_bursts.wav"), bursts, args.sr)
    
    # Stereo panning
    panning = generate_stereo_panning(args.duration, args.sr)
    torchaudio.save(os.path.join(args.output, "stereo_panning.wav"), panning, args.sr)
    
    # Complex mix
    mix = generate_complex_mix(args.duration, args.sr)
    torchaudio.save(os.path.join(args.output, "complex_mix.wav"), mix, args.sr)
    
    print(f"Done! Test audio files saved to {args.output}/")
    print("Generated files:")
    print("  - sine_sweep.wav: Logarithmic frequency sweep")
    print("  - harmonic_tones.wav: Tone with harmonic overtones")
    print("  - noise_bursts.wav: Periodic noise bursts")
    print("  - stereo_panning.wav: Tone panning between left and right channels")
    print("  - complex_mix.wav: Complex mix of tones, noise, and transients")
    
if __name__ == "__main__":
    main() 