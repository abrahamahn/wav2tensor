"""
Analysis script for Wav2TensorLite, demonstrating the early fusion approach.
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import time
from datetime import datetime

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2tensor.core import Wav2TensorCore
from wav2tensor.core_lite import Wav2TensorLite

def compare_representations(
    audio_file,
    output_dir='results',
    start_time=0,
    duration=5,
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    bit_depth=16,
    fusion_method='concat',
    harmonic_method='hps',
    include_planes=None,
    use_adaptive_freq=True
):
    """
    Compare original Wav2Tensor with the new Wav2TensorLite.
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory to save outputs
        start_time: Start time in seconds
        duration: Duration in seconds
        sample_rate: Audio sample rate
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        bit_depth: Bit depth for quantization (8 or 16)
        fusion_method: Method to fuse planes ('concat', 'add', 'learned')
        harmonic_method: 'hps' or 'filterbank'
        include_planes: List of planes to include
        use_adaptive_freq: Whether to use frequency-adaptive resolution
    
    Returns:
        Dictionary with analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio file
    print(f"Loading audio: {audio_file}")
    waveform, file_sample_rate = torchaudio.load(audio_file)
    
    # Resample if needed
    if file_sample_rate != sample_rate:
        print(f"Resampling from {file_sample_rate}Hz to {sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)
    
    # Calculate samples for the segment
    start_sample = int(start_time * sample_rate)
    segment_samples = int(duration * sample_rate)
    
    # Extract the segment
    end_sample = min(start_sample + segment_samples, waveform.shape[1])
    segment = waveform[:, start_sample:end_sample]
    segment_duration = (end_sample - start_sample) / sample_rate
    
    print(f"Analyzing segment from {start_time:.1f}s to {start_time + segment_duration:.1f}s ({segment_duration:.1f}s duration)")
    print(f"Segment shape: {segment.shape}")
    
    # Add batch dimension
    segment = segment.unsqueeze(0)  # [B, C, T]
    
    # Create filename prefix for outputs
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    segment_info = f"_{int(start_time)}s_to_{int(start_time + segment_duration)}s"
    output_prefix = f"{output_dir}/{base_name}{segment_info}"
    
    # Initialize timer
    start_time_original = time.time()
    
    # Initialize original Wav2Tensor
    print(f"Initializing original Wav2TensorCore...")
    wav2tensor_original = Wav2TensorCore(
        sample_rate=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        harmonic_method=harmonic_method,
        include_planes=include_planes
    )
    
    # Process with original Wav2Tensor
    tensor_original, planes_original = wav2tensor_original(segment)
    process_time_original = time.time() - start_time_original
    print(f"Original Wav2Tensor processed in {process_time_original:.2f} seconds")
    
    # Initialize timer for Lite version
    start_time_lite = time.time()
    
    # Initialize Wav2TensorLite
    print(f"Initializing Wav2TensorLite with fusion_method='{fusion_method}', bit_depth={bit_depth}...")
    wav2tensor_lite = Wav2TensorLite(
        sample_rate=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        bit_depth=bit_depth,
        use_adaptive_freq=use_adaptive_freq,
        harmonic_method=harmonic_method,
        include_planes=include_planes,
        fusion_method=fusion_method
    )
    
    # Process with Wav2TensorLite
    tensor_lite, metadata_lite = wav2tensor_lite(segment)
    process_time_lite = time.time() - start_time_lite
    print(f"Wav2TensorLite processed in {process_time_lite:.2f} seconds")
    print(f"Speedup factor: {process_time_original / process_time_lite:.2f}x")
    
    # Compare memory usage
    memory_original_kb = tensor_original.element_size() * tensor_original.nelement() / 1024
    memory_lite_kb = tensor_lite.element_size() * tensor_lite.nelement() / 1024
    print(f"Memory usage - Original: {memory_original_kb:.2f} KB, Lite: {memory_lite_kb:.2f} KB")
    print(f"Memory reduction: {memory_original_kb / memory_lite_kb:.2f}x")
    
    # Collect results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_file": audio_file,
        "segment": {
            "start": float(start_time),
            "duration": float(segment_duration),
            "shape": segment.shape[1:],  # Exclude batch dimension
        },
        "settings": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "harmonic_method": harmonic_method,
            "include_planes": include_planes,
            "bit_depth": bit_depth,
            "use_adaptive_freq": use_adaptive_freq,
            "fusion_method": fusion_method
        },
        "performance": {
            "original_time": process_time_original,
            "lite_time": process_time_lite,
            "speedup": process_time_original / process_time_lite,
            "original_memory_kb": float(memory_original_kb),
            "lite_memory_kb": float(memory_lite_kb),
            "memory_reduction": float(memory_original_kb / memory_lite_kb)
        },
        "tensor_shapes": {
            "original": list(tensor_original.shape),
            "lite": list(tensor_lite.shape)
        }
    }
    
    # Generate visualizations
    visualize_comparison(
        segment[0],  # Remove batch dimension for visualization
        planes_original,
        tensor_lite,
        metadata_lite, 
        output_prefix,
        sample_rate,
        n_fft
    )
    
    return results

def visualize_comparison(waveform, original_planes, lite_tensor, lite_metadata, output_prefix, sample_rate, n_fft):
    """Generate visualizations to compare original and lite representations"""
    # Create figure with multiple plots
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Wav2Tensor vs Wav2TensorLite Comparison", fontsize=16)
    
    # Calculate frequency axis in Hz
    if lite_metadata.get('use_adaptive_freq', False):
        # For adaptive frequency, use the values from metadata
        freqs = np.array(lite_metadata.get('adaptive_freqs', 
                                        np.linspace(0, sample_rate/2, n_fft//2 + 1)))
    else:
        freqs = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
    # Add meaningful frequency ticks for different ranges
    freq_ticks_full = [20, 100, 500, 1000, 2000, 5000, 10000]
    freq_ticks_low = [20, 50, 100, 200, 300, 500, 750, 1000]
    
    # Calculate time axis
    spec_mag = torch.abs(original_planes['spectral']).squeeze().detach().cpu().numpy()
    time_frames = spec_mag.shape[1]
    hop_length = n_fft // 4  # Assuming default hop_length = n_fft/4
    duration_sec = time_frames * hop_length / sample_rate
    time_ticks = np.linspace(0, time_frames, 5)
    time_labels = [f"{t:.1f}" for t in np.linspace(0, duration_sec, 5)]
    
    # Plot original audio waveform
    axs[0, 0].plot(np.arange(waveform.shape[1]) / sample_rate, waveform[0].detach().cpu().numpy())
    axs[0, 0].set_title("Audio Waveform")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")
    
    # If stereo, plot both channels
    if waveform.shape[0] == 2:
        axs[0, 0].plot(np.arange(waveform.shape[1]) / sample_rate, waveform[1].detach().cpu().numpy(), alpha=0.7)
        axs[0, 0].legend(['Left', 'Right'])
    
    # Plot original spectral magnitude
    spec_mag_db = 10 * np.log10(spec_mag + 1e-8)
    im1 = axs[0, 1].imshow(
        spec_mag_db, 
        aspect='auto', 
        origin='lower',
        extent=[0, spec_mag.shape[1], 0, sample_rate/2]
    )
    axs[0, 1].set_title("Original Spectral Magnitude (dB)")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_yticks(freq_ticks_full)
    axs[0, 1].set_yticklabels([f"{f}" for f in freq_ticks_full])
    axs[0, 1].set_xticks(time_ticks)
    axs[0, 1].set_xticklabels(time_labels)
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot original harmonic plane
    if 'harmonic' in original_planes:
        harm = original_planes['harmonic'].squeeze().detach().cpu().numpy()
        # Use log1p visualization for better dynamic range
        log_harm = np.log1p(harm)
        im2 = axs[0, 2].imshow(
            log_harm, 
            aspect='auto', 
            origin='lower',
            extent=[0, harm.shape[1], 0, sample_rate/2]
        )
        axs[0, 2].set_title("Original Harmonic Plane (log1p)")
        axs[0, 2].set_ylabel("Frequency (Hz)")
        axs[0, 2].set_yticks(freq_ticks_full)
        axs[0, 2].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[0, 2].set_xticks(time_ticks)
        axs[0, 2].set_xticklabels(time_labels)
        plt.colorbar(im2, ax=axs[0, 2])
    
    # Plot original spatial plane - IPD
    if 'spatial' in original_planes and original_planes['spatial'].shape[1] >= 1:
        spatial = original_planes['spatial'].squeeze().detach().cpu().numpy()
        ipd = spatial[0]
        im3 = axs[1, 0].imshow(
            ipd, 
            aspect='auto', 
            origin='lower',
            extent=[0, ipd.shape[1], 0, sample_rate/2],
            cmap='hsv',
            vmin=-np.pi, 
            vmax=np.pi
        )
        axs[1, 0].set_title("Original Spatial - IPD")
        axs[1, 0].set_ylabel("Frequency (Hz)")
        axs[1, 0].set_yticks(freq_ticks_full)
        axs[1, 0].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[1, 0].set_xticks(time_ticks)
        axs[1, 0].set_xticklabels(time_labels)
        plt.colorbar(im3, ax=axs[1, 0])
    
    # Plot original spatial plane - Energy Panning
    if 'spatial' in original_planes and original_planes['spatial'].shape[1] >= 2:
        energy = spatial[1]
        im4 = axs[1, 1].imshow(
            energy, 
            aspect='auto', 
            origin='lower',
            extent=[0, energy.shape[1], 0, sample_rate/2],
            cmap='coolwarm',
            vmin=-1, 
            vmax=1
        )
        axs[1, 1].set_title("Original Spatial - Energy Panning")
        axs[1, 1].set_ylabel("Frequency (Hz)")
        axs[1, 1].set_yticks(freq_ticks_full)
        axs[1, 1].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[1, 1].set_xticks(time_ticks)
        axs[1, 1].set_xticklabels(time_labels)
        plt.colorbar(im4, ax=axs[1, 1])
    
    # Plot original psychoacoustic plane
    if 'psychoacoustic' in original_planes:
        psycho = original_planes['psychoacoustic'].squeeze().detach().cpu().numpy()
        im5 = axs[1, 2].imshow(
            psycho, 
            aspect='auto', 
            origin='lower',
            extent=[0, psycho.shape[1], 0, sample_rate/2],
            vmin=0, 
            vmax=1
        )
        axs[1, 2].set_title("Original Psychoacoustic Plane")
        axs[1, 2].set_ylabel("Frequency (Hz)")
        axs[1, 2].set_yticks(freq_ticks_full)
        axs[1, 2].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[1, 2].set_xticks(time_ticks)
        axs[1, 2].set_xticklabels(time_labels)
        plt.colorbar(im5, ax=axs[1, 2])
    
    # Plot the first 3 channels of the fused lite tensor
    lite_np = lite_tensor.squeeze().detach().cpu().numpy()
    for i in range(min(3, lite_tensor.shape[1])):
        im_lite = axs[2, i].imshow(
            lite_np[i], 
            aspect='auto', 
            origin='lower',
            extent=[0, lite_np.shape[1], 0, sample_rate/2]
        )
        axs[2, i].set_title(f"Lite Fused Channel {i+1}")
        axs[2, i].set_ylabel("Frequency (Hz)")
        axs[2, i].set_yticks(freq_ticks_full)
        axs[2, i].set_yticklabels([f"{f}" for f in freq_ticks_full])
        axs[2, i].set_xticks(time_ticks)
        axs[2, i].set_xticklabels(time_labels)
        plt.colorbar(im_lite, ax=axs[2, i])
    
    # Add fusion method and stats to plot
    stats_text = f"Fusion Method: {lite_metadata.get('fusion_method', 'concat')}\n"
    stats_text += f"Bit Depth: {lite_metadata.get('bit_depth', 16)}\n"
    stats_text += f"Adaptive Freq: {lite_metadata.get('use_adaptive_freq', True)}\n"
    stats_text += f"Original Shape: {lite_metadata.get('tensor_shape', {}).get('spectral', 'N/A')}\n"
    stats_text += f"Fused Shape: {lite_metadata.get('tensor_shape', {}).get('fused', 'N/A')}"
    
    # Find an empty subplot if available
    for i in range(2, 3):
        for j in range(0, 3):
            if i == 2 and j >= min(3, lite_tensor.shape[1]):
                axs[i, j].text(0.5, 0.5, stats_text, ha='center', va='center', transform=axs[i, j].transAxes)
                axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the visualization
    viz_path = f"{output_prefix}_comparison.png"
    plt.savefig(viz_path, dpi=300)
    print(f"Comparison visualization saved to {viz_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Compare Wav2Tensor and Wav2TensorLite")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0s)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (default: 5s)")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[8, 16], help="Bit depth for quantization (8 or 16)")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "add", "learned"], 
                        help="Fusion method (default: concat)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive frequency resolution")
    parser.add_argument("--method", type=str, default="hps", choices=["hps", "filterbank"], 
                        help="Harmonic plane method (default: hps)")
    parser.add_argument("--planes", type=str, default="default", 
                        help="Planes to include, comma-separated (default: 'default' for spectral,harmonic,spatial)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    
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
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run the comparison
        results = compare_representations(
            audio_file=args.audio_file,
            output_dir=args.output_dir,
            start_time=args.start,
            duration=args.duration,
            bit_depth=args.bit_depth,
            fusion_method=args.fusion,
            harmonic_method=args.method,
            include_planes=include_planes,
            use_adaptive_freq=not args.no_adaptive
        )
        
        # Save results to file
        import json
        results_path = f"{args.output_dir}/comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comparison results saved to {results_path}")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 