"""
Reconstruction Quality Test for audio representations.

This script evaluates how well different audio representations preserve 
audio information by measuring reconstruction quality:

1. Load audio samples
2. Convert to representation (Wav2Tensor, mel-spectrogram, raw waveform)
3. Convert back to audio
4. Measure reconstruction quality using objective metrics

The test isolates representation quality from model performance.
"""

import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import signal
from concurrent.futures import ThreadPoolExecutor
import librosa

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import wav2tensor
from wav2tensor.core import Wav2TensorCore

# Import our custom metrics (to avoid external dependencies)
def calculate_mse(clean, enhanced):
    """Calculate mean squared error between signals"""
    return np.mean((clean - enhanced) ** 2)

def calculate_snr(clean, enhanced):
    """Calculate signal-to-noise ratio in dB"""
    noise = clean - enhanced
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return 100.0  # Very high SNR if noise is zero

def calculate_lsd(clean_spec, enhanced_spec):
    """Calculate Log-Spectral Distance
    Takes magnitude spectrograms as input"""
    # Add small value to avoid log(0)
    eps = 1e-8
    clean_log = np.log10(np.abs(clean_spec) + eps)
    enhanced_log = np.log10(np.abs(enhanced_spec) + eps)
    
    # Compute mean squared difference in log domain
    return np.mean(np.sqrt(np.mean((clean_log - enhanced_log) ** 2, axis=0)))

class BaseProcessor:
    """Base class for audio representation processors."""
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def forward(self, waveform):
        """Convert waveform to representation."""
        raise NotImplementedError
    
    def inverse(self, representation):
        """Convert representation back to waveform."""
        raise NotImplementedError

class IdentityProcessor(BaseProcessor):
    """Raw waveform processor (identity transformation)"""
    def __init__(self, sample_rate=22050):
        super().__init__(sample_rate)
    
    def forward(self, waveform):
        """Convert waveform to 'representation' (no change)"""
        return waveform
    
    def inverse(self, representation):
        """Convert 'representation' back to waveform (no change)"""
        return representation

class STFTProcessor(BaseProcessor):
    """STFT processor (complex spectrogram)"""
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256):
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Create a window and store it for consistent use
        self.window = torch.hann_window(n_fft)
        
    def forward(self, waveform):
        """Convert waveform to complex STFT"""
        return torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
    
    def inverse(self, complex_spec):
        """Convert complex STFT back to waveform"""
        return torch.istft(
            complex_spec, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            window=self.window,
            length=None  # Let PyTorch determine the length
        )

class MelSpectrogramProcessor(BaseProcessor):
    """Mel-spectrogram processor"""
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2  # Use power spectrogram
        )
        
        # Get the mel basis for later use in inverse
        self.mel_basis = self.mel_spec.mel_scale.fb
        
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2,  # Match power spectrogram
            n_iter=32  # Increase iterations for better reconstruction
        )
    
    def forward(self, waveform):
        """Convert waveform to mel-spectrogram"""
        # Add small epsilon to avoid log(0)
        mel_spec = self.mel_spec(waveform)
        return torch.log1p(mel_spec)  # Use log1p for better numerical stability
    
    def inverse(self, mel_spec):
        """Convert mel-spectrogram back to waveform using Griffin-Lim
        
        We need to convert the mel spectrogram back to a linear spectrogram
        before applying Griffin-Lim algorithm.
        """
        # Convert back from log domain
        mel_spec = torch.exp(mel_spec) - 1.0  # Inverse of log1p
        
        # Ensure positive values
        mel_spec = torch.clamp(mel_spec, min=0.0)
        
        # Create an approximate linear spectrogram from mel spectrogram
        # We use pseudo-inverse of the mel basis
        mel_to_linear = torch.pinverse(self.mel_basis)
        linear_spec = torch.matmul(mel_spec.transpose(-1, -2), mel_to_linear).transpose(-1, -2)
        
        # Ensure positive values again after transformation
        linear_spec = torch.clamp(linear_spec, min=0.0)
        
        # Since this is magnitude only, we need to use Griffin-Lim to recover phase
        try:
            return self.griffin_lim(linear_spec)
        except Exception as e:
            print(f"Error in Griffin-Lim: {str(e)}")
            # Return silence of appropriate length if reconstruction fails
            expected_length = (linear_spec.shape[-1] - 1) * self.hop_length
            return torch.zeros((1, expected_length), device=linear_spec.device)

class Wav2TensorProcessor(BaseProcessor):
    """Wav2Tensor representation processor."""
    
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256,
                 harmonic_method='hps', include_planes=None,
                 use_adaptive_freq=True, target_freq_bins=256):
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_method = harmonic_method
        self.include_planes = include_planes
        self.use_adaptive_freq = use_adaptive_freq
        self.target_freq_bins = target_freq_bins
        
        # Initialize Wav2Tensor core
        self.core = Wav2TensorCore(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            harmonic_method=harmonic_method,
            include_planes=include_planes,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins
        )
    
    def forward(self, waveform):
        """Convert waveform to Wav2Tensor representation."""
        # Ensure waveform is 3D: [batch, channels, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        # Process with Wav2Tensor core
        tensor, self.planes = self.core(waveform)
        return tensor
    
    def inverse(self, representation):
        """Convert Wav2Tensor representation back to waveform."""
        # Use the stored planes if available
        planes = getattr(self, 'planes', None)
        
        # Convert back using Wav2Tensor core
        return self.core.inverse(representation, planes)

def evaluate_reconstruction(processor, audio_file, output_dir=None, sample_rate=22050,
                           segment_length=None, segment_start=0):
    """
    Evaluate reconstruction quality for a given processor.
    
    Args:
        processor: Audio processor instance
        audio_file: Path to audio file
        output_dir: Directory to save results (optional)
        sample_rate: Target sample rate
        segment_length: Length of segment to process (in samples), None or 0 for full length
        segment_start: Start position for segment (in samples)
        
    Returns:
        Dictionary with reconstruction quality metrics
    """
    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_file)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Extract segment if specified and valid
        if segment_length and segment_length > 0:
            if segment_start + segment_length > waveform.shape[1]:
                segment_start = 0  # Reset to start if segment would exceed length
            waveform = waveform[:, segment_start:segment_start + segment_length]
        
        # Ensure audio is finite
        if not torch.isfinite(waveform).all():
            raise ValueError("Input audio contains NaN or Inf values")
        
        # Process forward and backward
        with torch.no_grad():  # Disable gradient computation for efficiency
            # Convert to representation
            representation = processor.forward(waveform)
            
            # Convert back to waveform
            reconstructed = processor.inverse(representation)
            
            # Ensure same length by padding/truncating
            if reconstructed.shape[-1] > waveform.shape[-1]:
                reconstructed = reconstructed[..., :waveform.shape[-1]]
            elif reconstructed.shape[-1] < waveform.shape[-1]:
                reconstructed = F.pad(reconstructed, (0, waveform.shape[-1] - reconstructed.shape[-1]))
            
            # Ensure output is finite
            if not torch.isfinite(reconstructed).all():
                raise ValueError("Reconstructed audio contains NaN or Inf values")
        
        # Convert to numpy for metrics calculation
        waveform = waveform.cpu().numpy()
        reconstructed = reconstructed.squeeze(0).cpu().numpy()  # Remove batch dimension but keep channels
        
        # Calculate metrics per channel and average
        metrics = {
            'mse': [],
            'snr': [],
            'lsd': []
        }
        
        n_channels = waveform.shape[0]
        for ch in range(n_channels):
            # Get channel data
            waveform_ch = waveform[ch]
            reconstructed_ch = reconstructed[ch] if n_channels > 1 else reconstructed
            
            # Calculate metrics for this channel
            mse = calculate_mse(waveform_ch, reconstructed_ch)
            snr = calculate_snr(waveform_ch, reconstructed_ch)
            
            # Calculate LSD using STFT
            n_fft = 512  # Use smaller FFT size for efficiency
            hop_length = 128
            
            # Compute STFTs efficiently
            stft_orig = np.abs(librosa.stft(waveform_ch, n_fft=n_fft, hop_length=hop_length))
            stft_recon = np.abs(librosa.stft(reconstructed_ch, n_fft=n_fft, hop_length=hop_length))
            lsd = calculate_lsd(stft_orig, stft_recon)
            
            # Store metrics
            metrics['mse'].append(mse)
            metrics['snr'].append(snr)
            metrics['lsd'].append(lsd)
        
        # Average metrics across channels
        avg_metrics = {
            'mse': float(np.mean(metrics['mse'])),
            'snr': float(np.mean(metrics['snr'])),
            'lsd': float(np.mean(metrics['lsd']))
        }
        
        # Create visualization data if output directory is specified
        plot_data = None
        if output_dir is not None:
            # Save reconstructed audio
            reconstructed_path = os.path.join(output_dir, "reconstructed.wav")
            torchaudio.save(
                reconstructed_path,
                torch.tensor(reconstructed[None, :] if n_channels == 1 else reconstructed),  # Add batch dim for mono
                sample_rate
            )
            
            # Plot full waveform comparison
            plot_data = {
                'waveform_orig': waveform,
                'waveform_recon': reconstructed,
                'stft_diff': [np.log(np.abs(librosa.stft(waveform[ch] - reconstructed[ch] if n_channels > 1 else reconstructed, n_fft=n_fft, hop_length=hop_length)) + 1e-8) for ch in range(n_channels)]
            }
        
        # Free memory explicitly
        del representation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Validate metrics
        if not all(np.isfinite([avg_metrics['mse'], avg_metrics['snr'], avg_metrics['lsd']])):
            raise ValueError("Invalid metric values computed")
        
        return {
            'mse': avg_metrics['mse'],
            'snr': avg_metrics['snr'],
            'lsd': avg_metrics['lsd'],
            'plot_data': plot_data,
            'sample_rate': sample_rate,
            'n_channels': n_channels
        }
    except Exception as e:
        print(f"Error in evaluate_reconstruction: {str(e)}")
        raise  # Re-raise to be handled by process_single

def create_plots(results, output_dir, audio_name):
    """Create plots in the main thread."""
    import matplotlib.pyplot as plt
    
    # Get sample rate and number of channels
    sample_rate = results.get('sample_rate', 22050)
    n_channels = results.get('n_channels', 1)
    
    # Create visualization plots
    plt.figure(figsize=(15, 5 * n_channels))
    
    # Plot waveforms for each channel
    for ch in range(n_channels):
        plt.subplot(n_channels, 2, ch*2 + 1)
        time = np.arange(len(results['plot_data']['waveform_orig'][ch])) / sample_rate
        plt.plot(time, results['plot_data']['waveform_orig'][ch], label='Original', alpha=0.7)
        plt.plot(time, results['plot_data']['waveform_recon'][ch], label='Reconstructed', alpha=0.7)
        plt.title(f'Channel {ch+1} Waveform Comparison')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # Plot spectrogram difference
        plt.subplot(n_channels, 2, ch*2 + 2)
        plt.title(f'Channel {ch+1} Spectrogram Difference')
        plt.imshow(
            results['plot_data']['stft_diff'][ch],
            aspect='auto',
            origin='lower',
            extent=[0, time[-1], 0, sample_rate/2]
        )
        plt.colorbar(label='Log Difference')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{audio_name}_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def process_single(processor_info):
    """Process a single audio file with a given processor."""
    processor_name, processor, audio_file, audio_name, output_dir, sample_rate, segment_length = processor_info
    
    print(f"\nProcessing {audio_name} with {processor_name}...")
    
    # Create processor-specific output directory
    processor_dir = os.path.join(output_dir, processor_name, audio_name)
    os.makedirs(processor_dir, exist_ok=True)
    
    try:
        # Process directly
        result = evaluate_reconstruction(
            processor,
            audio_file,
            output_dir=processor_dir,
            sample_rate=sample_rate,
            segment_length=segment_length
        )
        
        # Validate metrics
        if not isinstance(result, dict) or not all(k in result for k in ['mse', 'snr', 'lsd']):
            raise ValueError("Invalid result format")
            
        if any(not np.isfinite(result[k]) for k in ['mse', 'snr', 'lsd']):
            raise ValueError("Invalid metric values")
        
        # Store plot data for later processing in main thread
        plot_data = result.pop('plot_data', None)
        
        return processor_name, result, plot_data
    except Exception as e:
        print(f"ERROR processing {processor_name}: {str(e)}")
        # Return valid but error-indicating values
        return processor_name, {
            'mse': 1.0,  # High MSE indicates poor reconstruction
            'snr': -20.0,  # Low SNR indicates poor reconstruction
            'lsd': 10.0,  # High LSD indicates poor reconstruction
            'error': str(e)
        }, None

def run_reconstruction_test(audio_files, output_dir, sample_rate=22050, segment_duration=1.0, batch_size=4):
    """
    Run reconstruction test on multiple audio files with different processors.
    
    Args:
        audio_files: List of audio file paths
        output_dir: Directory to save results
        sample_rate: Target sample rate
        segment_duration: Duration of segment to process in seconds (default: 1.0s, 0 for full length)
        batch_size: Number of processors to run in parallel
        
    Returns:
        Dictionary with test results for all processors and files
    """
    print("\nStarting reconstruction test...")
    segment_length = int(segment_duration * sample_rate) if segment_duration > 0 else None
    print(f"Using {'full audio length' if segment_length is None else f'{segment_length} samples'}")
    
    # Define processors to test
    print("\nInitializing processors...")
    processors = {
        'raw_waveform': IdentityProcessor(sample_rate=sample_rate),
        'stft': STFTProcessor(
            sample_rate=sample_rate,
            n_fft=1024,  # Using standard FFT size
            hop_length=256  # Using standard hop length
        ),
        'mel_spectrogram': MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_fft=1024,  # Using standard FFT size
            hop_length=256,  # Using standard hop length
            n_mels=80
        ),
        'wav2tensor': Wav2TensorProcessor(
            sample_rate=sample_rate,
            n_fft=1024,  # Using standard FFT size
            hop_length=256,  # Using standard hop length
            harmonic_method='hps',
            include_planes=['spectral', 'harmonic', 'spatial', 'psychoacoustic'],  # Using all planes
            use_adaptive_freq=False,  # Disable adaptive frequency for now
            target_freq_bins=None
        )
    }
    print("Processors initialized.")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Process each audio file with each processor
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        print(f"\nProcessing file: {audio_file}")
        audio_name = os.path.splitext(os.path.basename(audio_file))[0]
        all_results[audio_name] = {}
        
        # Create a thread pool for parallel processing
        from concurrent.futures import ThreadPoolExecutor
        
        # Process processors in parallel batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process_single, (name, proc, audio_file, audio_name, output_dir, sample_rate, segment_length)) 
                      for name, proc in processors.items()]
            
            # Collect results as they complete
            for future in futures:
                processor_name, result, plot_data = future.result()
                all_results[audio_name][processor_name] = result
                
                # Create plots in main thread if we have plot data
                if plot_data is not None:
                    processor_dir = os.path.join(output_dir, processor_name, audio_name)
                    create_plots({'plot_data': plot_data}, processor_dir, audio_name)
                
                print(f"Completed {processor_name} processing.")
    
    # Compute summary statistics across all files
    print("\nComputing summary statistics...")
    summary = {processor_name: {metric: 0 for metric in ['mse', 'snr', 'lsd']} 
               for processor_name in processors.keys()}
    
    # Average metrics across all audio files
    for audio_name, audio_results in all_results.items():
        for processor_name, processor_results in audio_results.items():
            if 'error' not in processor_results:
                for metric in ['mse', 'snr', 'lsd']:
                    summary[processor_name][metric] += processor_results[metric]
    
    # Calculate averages
    num_files = len(audio_files)
    for processor_name in processors.keys():
        for metric in ['mse', 'snr', 'lsd']:
            summary[processor_name][metric] /= num_files
    
    # Save all results to JSON
    print("\nSaving results...")
    with open(os.path.join(output_dir, 'reconstruction_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary to JSON
    with open(os.path.join(output_dir, 'reconstruction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create summary visualizations
    print("\nCreating visualizations...")
    create_summary_plots(summary, output_dir)
    
    print("\nReconstruction test completed.")
    return all_results, summary

def create_summary_plots(summary, output_dir):
    """Create summary plots for reconstruction test."""
    # Get processor names and extract metrics
    processor_names = list(summary.keys())
    mse_values = [summary[name]['mse'] for name in processor_names]
    snr_values = [summary[name]['snr'] for name in processor_names]
    lsd_values = [summary[name]['lsd'] for name in processor_names]
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # MSE plot (lower is better)
    ax1.bar(processor_names, mse_values)
    ax1.set_title('Mean Squared Error (lower is better)')
    ax1.set_ylabel('MSE')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # SNR plot (higher is better)
    ax2.bar(processor_names, snr_values)
    ax2.set_title('Signal-to-Noise Ratio (higher is better)')
    ax2.set_ylabel('SNR (dB)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # LSD plot (lower is better)
    ax3.bar(processor_names, lsd_values)
    ax3.set_title('Log-Spectral Distance (lower is better)')
    ax3.set_ylabel('LSD')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_summary.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio representation reconstruction quality")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file or directory")
    parser.add_argument("--output", type=str, default="results/reconstruction_test", 
                        help="Output directory for results")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate to use")
    parser.add_argument("--duration", type=float, default=0.5,  # Reduced from 5.0 to 0.5 seconds
                        help="Duration of segment to process in seconds")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of processors to run in parallel")
    
    args = parser.parse_args()
    
    try:
        # Find audio files
        if args.audio is None:
            # Use built-in audio files (if we had any)
            print("No audio files specified, looking for WAV files in current directory...")
            audio_files = glob("*.wav")
            if not audio_files:
                print("No WAV files found. Please specify an audio file or directory.")
                return
        elif os.path.isdir(args.audio):
            # Use all WAV files in the specified directory
            audio_files = glob(os.path.join(args.audio, "*.wav"))
        else:
            # Use the specified audio file
            if not os.path.exists(args.audio):
                print(f"Audio file not found: {args.audio}")
                return
            audio_files = [args.audio]
        
        # Ensure we found audio files
        if not audio_files:
            print("No audio files found. Please specify a valid audio file or directory.")
            return
        
        print(f"Found {len(audio_files)} audio file(s) to process:")
        for audio_file in audio_files:
            print(f"  - {audio_file}")
        
        # Run reconstruction test
        results, summary = run_reconstruction_test(
            audio_files,
            args.output,
            sample_rate=args.sr,
            segment_duration=args.duration,
            batch_size=args.batch_size
        )
        
        # Print summary
        print("\nReconstruction Test Summary:")
        for processor_name, metrics in summary.items():
            print(f"\n{processor_name}:")
            print(f"  MSE: {metrics['mse']:.4f} (lower is better)")
            print(f"  SNR: {metrics['snr']:.4f} dB (higher is better)")
            print(f"  LSD: {metrics['lsd']:.4f} (lower is better)")
        
        print(f"\nResults saved to {args.output}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 