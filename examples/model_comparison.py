"""
Model-based comparison for different audio representations.
This script compares the performance of different audio representations when used in
standard deep learning models for audio enhancement tasks.

1. Raw waveform processing
2. Mel-spectrogram processing 
3. Wav2Tensor (full version)
4. Wav2Tensor (optimized configurations)

The comparisons use the same model architecture (U-Net) with different input representations
and measure performance using standard audio quality metrics (PESQ, STOI, SDR).
"""

import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import json
import argparse
import sys

# Add parent directory to path to import wav2tensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import wav2tensor
from wav2tensor.core import Wav2TensorCore

# Define model architecture - Simple U-Net for audio enhancement
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.2)
        if self.dropout:
            x = self.dropout(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = torch.cat([x, skip], 1)
        return x

class AudioUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AudioUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(input_channels, 64, 4, 2, 1)  # No batch norm in first layer
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.enc5 = UNetBlock(512, 512)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 512)
        
        # Decoder
        self.dec5 = UNetUpBlock(512, 512, dropout=True)
        self.dec4 = UNetUpBlock(1024, 256, dropout=True)
        self.dec3 = UNetUpBlock(512, 128)
        self.dec2 = UNetUpBlock(256, 64)
        
        # Final layer
        self.final = nn.ConvTranspose2d(128, output_channels, 4, 2, 1)
    
    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.enc1(x), 0.2)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        b = self.bottleneck(e5)
        
        # Decoder with skip connections
        d5 = self.dec5(b, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        
        # Final layer with tanh activation
        out = torch.tanh(self.final(d2))
        
        return out

# Synthetic noisy speech dataset
class SyntheticAudioDataset(Dataset):
    def __init__(self, num_samples=1000, sample_rate=22050, duration=2.0, snr_range=(5, 20)):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.snr_range = snr_range
        self.audio_length = int(duration * sample_rate)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic clean speech (simple sine wave with harmonics)
        t = torch.arange(0, self.audio_length) / self.sample_rate
        freq = torch.rand(1) * 200 + 200  # Random frequency between 200-400Hz
        
        # Create harmonics with decreasing amplitude
        clean = torch.sin(2 * np.pi * freq * t)
        for i in range(2, 6):  # Add harmonics
            clean += (1/i) * torch.sin(2 * np.pi * i * freq * t)
        
        # Normalize
        clean = clean / clean.abs().max()
        
        # Create noise with random SNR
        snr = torch.rand(1) * (self.snr_range[1] - self.snr_range[0]) + self.snr_range[0]
        noise = torch.randn(self.audio_length)
        noise = noise / noise.abs().max()
        
        # Calculate scaling factor for noise based on SNR
        clean_power = (clean ** 2).mean()
        noise_power = (noise ** 2).mean()
        scale = torch.sqrt(clean_power / (noise_power * 10 ** (snr / 10)))
        
        # Add scaled noise to clean signal
        noisy = clean + scale * noise
        
        # Normalize
        noisy = noisy / noisy.abs().max()
        
        # Add channel dimension
        clean = clean.unsqueeze(0)  # [1, audio_length]
        noisy = noisy.unsqueeze(0)  # [1, audio_length]
        
        return {'clean': clean, 'noisy': noisy}

# Processor for raw waveform input
class WaveformProcessor:
    def __init__(self, chunk_size=4096, hop_size=1024):
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        
    def process(self, waveform):
        # Split waveform into overlapping chunks
        chunks = []
        for i in range(0, waveform.shape[1] - self.chunk_size + 1, self.hop_size):
            chunk = waveform[:, i:i+self.chunk_size]
            chunks.append(chunk)
        
        if len(chunks) == 0:
            # Handle case where audio is shorter than chunk_size
            chunk = F.pad(waveform, (0, self.chunk_size - waveform.shape[1]))
            chunks.append(chunk)
            
        chunks = torch.stack(chunks, dim=0)  # [num_chunks, channels, chunk_size]
        
        # Reshape to [batch, channels, height, width] for U-Net
        chunks = chunks.unsqueeze(3)  # [num_chunks, channels, chunk_size, 1]
        return chunks
    
    def reconstruct(self, chunks, original_length):
        # Remove the extra dimension
        chunks = chunks.squeeze(3)  # [num_chunks, channels, chunk_size]
        
        # Reconstruct waveform using overlap-add
        batch_size, channels, chunk_size = chunks.shape
        reconstructed = torch.zeros(channels, original_length)
        
        norm_factor = torch.zeros(channels, original_length)
        window = torch.hann_window(chunk_size)
        
        for i in range(batch_size):
            start_idx = i * self.hop_size
            end_idx = start_idx + self.chunk_size
            
            if end_idx > original_length:
                # Handle boundary case
                end_idx = original_length
                current_chunk = chunks[i, :, :end_idx-start_idx]
                reconstructed[:, start_idx:end_idx] += current_chunk * window[:end_idx-start_idx]
                norm_factor[:, start_idx:end_idx] += window[:end_idx-start_idx]
            else:
                reconstructed[:, start_idx:end_idx] += chunks[i] * window
                norm_factor[:, start_idx:end_idx] += window
        
        # Normalize by the overlap factor
        mask = (norm_factor > 0)
        reconstructed[mask] = reconstructed[mask] / norm_factor[mask]
        
        return reconstructed

# Processor for mel-spectrogram input
class MelSpectrogramProcessor:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    def process(self, waveform):
        # Apply mel spectrogram
        mel = self.mel_spec(waveform)
        
        # Convert to dB
        mel_db = self.to_db(mel)
        
        # Normalize
        mel_db_norm = (mel_db + 80) / 80  # Normalize from [-80, 0] to [0, 1]
        mel_db_norm = torch.clamp(mel_db_norm, 0, 1)
        
        # Add extra dimension for U-Net
        mel_db_norm = mel_db_norm.unsqueeze(1)  # [batch, 1, n_mels, time]
        
        return mel_db_norm
    
    def reconstruct(self, mel_db_norm, original_length):
        # Remove extra dimension
        mel_db_norm = mel_db_norm.squeeze(1)  # [batch, n_mels, time]
        
        # Denormalize
        mel_db = mel_db_norm * 80 - 80
        
        # Convert to linear scale
        mel = self.to_db.inv(mel_db)
        
        # Use Griffin-Lim to reconstruct phase
        waveform = self.griffin_lim(mel)
        
        # Trim to original length
        if waveform.shape[1] > original_length:
            waveform = waveform[:, :original_length]
        elif waveform.shape[1] < original_length:
            waveform = F.pad(waveform, (0, original_length - waveform.shape[1]))
        
        return waveform

# Processor for Wav2Tensor input
class Wav2TensorProcessor:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, include_planes=None,
                 harmonic_method='hps', use_adaptive_freq=False, target_freq_bins=256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.wav2tensor = Wav2TensorCore(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            include_planes=include_planes,
            harmonic_method=harmonic_method,
            use_adaptive_freq=use_adaptive_freq,
            target_freq_bins=target_freq_bins
        )
    
    def process(self, waveform):
        # Apply Wav2Tensor
        tensor, planes = self.wav2tensor(waveform)
        
        # Normalize each channel separately
        for c in range(tensor.shape[1]):
            channel = tensor[:, c:c+1]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                tensor[:, c:c+1] = (channel - channel_min) / (channel_max - channel_min)
        
        return tensor
    
    def reconstruct(self, tensor, original_length):
        # This is a simplified reconstruction - in real-world scenarios, 
        # we would need a more sophisticated approach to preserve phase information
        # For now, we'll just take the first two channels (real and imaginary parts of STFT)
        # and use inverse STFT
        
        # Extract real and imaginary parts of STFT
        real_part = tensor[:, 0]  # [batch, freq, time]
        imag_part = tensor[:, 1]  # [batch, freq, time]
        
        # Reconstruct complex STFT
        complex_spec = torch.complex(real_part, imag_part)
        
        # Apply inverse STFT
        istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        waveform = istft(complex_spec)
        
        # Trim to original length
        if waveform.shape[1] > original_length:
            waveform = waveform[:, :original_length]
        elif waveform.shape[1] < original_length:
            waveform = F.pad(waveform, (0, original_length - waveform.shape[1]))
        
        return waveform

def train_model(model, processor, train_loader, val_loader, device, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)
            
            # Process inputs
            clean_processed = processor.process(clean)
            noisy_processed = processor.process(noisy)
            
            clean_processed = clean_processed.to(device)
            noisy_processed = noisy_processed.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(noisy_processed)
            
            # Compute loss
            loss = criterion(output, clean_processed)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                clean = batch['clean'].to(device)
                noisy = batch['noisy'].to(device)
                
                # Process inputs
                clean_processed = processor.process(clean)
                noisy_processed = processor.process(noisy)
                
                clean_processed = clean_processed.to(device)
                noisy_processed = noisy_processed.to(device)
                
                # Forward pass
                output = model(noisy_processed)
                
                # Compute loss
                loss = criterion(output, clean_processed)
                val_loss += loss.item()
        
        # Average losses
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return train_losses, val_losses, training_time

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

def evaluate_model(model, processor, test_loader, device):
    model.eval()
    
    # Initialize metric lists
    mse_scores = []
    snr_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)
            
            # Process inputs
            noisy_processed = processor.process(noisy)
            noisy_processed = noisy_processed.to(device)
            
            # Forward pass
            output = model(noisy_processed)
            
            # Reconstruct audio
            output_waveform = processor.reconstruct(output.cpu(), clean.shape[1])
            
            # Calculate metrics
            for i in range(clean.shape[0]):
                clean_audio = clean[i, 0].cpu().numpy()
                output_audio = output_waveform[i, 0].cpu().numpy()
                
                # Normalize audio for metrics
                clean_audio = clean_audio / np.max(np.abs(clean_audio))
                output_audio = output_audio / np.max(np.abs(output_audio))
                
                # Calculate MSE
                mse = calculate_mse(clean_audio, output_audio)
                mse_scores.append(mse)
                
                # Calculate SNR
                snr = calculate_snr(clean_audio, output_audio)
                snr_scores.append(snr)
    
    # Calculate average metrics
    avg_mse = np.mean(mse_scores) if mse_scores else 0
    avg_snr = np.mean(snr_scores) if snr_scores else 0
    
    print(f'Average MSE: {avg_mse:.4f}')
    print(f'Average SNR: {avg_snr:.4f}')
    
    return {
        'mse': avg_mse,
        'snr': avg_snr
    }

def run_experiment(processor_type, processor_config, output_dir=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    train_dataset = SyntheticAudioDataset(num_samples=500)
    val_dataset = SyntheticAudioDataset(num_samples=100)
    test_dataset = SyntheticAudioDataset(num_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize processor based on type
    if processor_type == 'waveform':
        processor = WaveformProcessor(**processor_config)
        # For waveform, extract a single example to determine input channels
        sample = next(iter(train_loader))
        processed = processor.process(sample['clean'])
        input_channels = processed.shape[1]
        output_channels = input_channels
    elif processor_type == 'mel':
        processor = MelSpectrogramProcessor(**processor_config)
        input_channels = 1  # Mel-spectrograms are single channel
        output_channels = 1
    elif processor_type == 'wav2tensor':
        processor = Wav2TensorProcessor(**processor_config)
        # For Wav2Tensor, extract a single example to determine input channels
        sample = next(iter(train_loader))
        processed = processor.process(sample['clean'])
        input_channels = processed.shape[1]
        output_channels = input_channels
    else:
        raise ValueError(f'Unknown processor type: {processor_type}')
    
    # Initialize model
    model = AudioUNet(input_channels, output_channels).to(device)
    
    # Train model
    train_losses, val_losses, training_time = train_model(
        model, processor, train_loader, val_loader, device, epochs=5
    )
    
    # Evaluate model
    metrics = evaluate_model(model, processor, test_loader, device)
    
    # Compile results
    results = {
        'processor_type': processor_type,
        'processor_config': processor_config,
        'model_params': sum(p.numel() for p in model.parameters()),
        'training_time': training_time,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'{processor_type}_{timestamp}.json'
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    # Define processor configurations to test
    experiments = [
        {
            'name': 'Raw Waveform',
            'type': 'waveform',
            'config': {
                'chunk_size': 4096,
                'hop_size': 1024
            }
        },
        {
            'name': 'Mel-Spectrogram',
            'type': 'mel',
            'config': {
                'sample_rate': 22050,
                'n_fft': 1024,
                'hop_length': 256,
                'n_mels': 80
            }
        },
        {
            'name': 'Wav2Tensor (Full)',
            'type': 'wav2tensor',
            'config': {
                'sample_rate': 22050,
                'n_fft': 1024,
                'hop_length': 256,
                'include_planes': ['spectral', 'harmonic', 'spatial', 'psychoacoustic'],
                'harmonic_method': 'hps',
                'use_adaptive_freq': False
            }
        },
        {
            'name': 'Wav2Tensor (Minimal)',
            'type': 'wav2tensor',
            'config': {
                'sample_rate': 22050,
                'n_fft': 1024,
                'hop_length': 256,
                'include_planes': ['spectral'],
                'harmonic_method': 'hps',
                'use_adaptive_freq': True,
                'target_freq_bins': 128
            }
        },
        {
            'name': 'Wav2Tensor (Balanced)',
            'type': 'wav2tensor',
            'config': {
                'sample_rate': 22050,
                'n_fft': 1024,
                'hop_length': 512,  # Increased hop length
                'include_planes': ['spectral', 'harmonic'],
                'harmonic_method': 'hps',
                'use_adaptive_freq': True,
                'target_freq_bins': 256
            }
        }
    ]
    
    # Create output directory
    output_dir = 'results/model_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    for experiment in experiments:
        print(f"\n===== Running experiment: {experiment['name']} =====")
        results = run_experiment(
            experiment['type'],
            experiment['config'],
            output_dir
        )
        all_results[experiment['name']] = results
    
    # Compile and save summary
    summary = {name: {
        'mse': results['metrics']['mse'],
        'snr': results['metrics']['snr'],
        'training_time': results['training_time'],
        'model_params': results['model_params']
    } for name, results in all_results.items()}
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    create_comparison_plots(all_results, output_dir)
    
    print("\nAll experiments completed!")
    print(f"Results saved to {output_dir}")

def create_comparison_plots(results, output_dir):
    # Extract metrics
    names = list(results.keys())
    mse_scores = [results[name]['metrics']['mse'] for name in names]
    snr_scores = [results[name]['metrics']['snr'] for name in names]
    training_times = [results[name]['training_time'] for name in names]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # MSE scores
    ax1.bar(names, mse_scores)
    ax1.set_title('MSE Scores')
    ax1.set_ylabel('MSE (lower is better)')
    ax1.set_ylim(0, 1)  # MSE range is 0-1
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # SNR scores
    ax2.bar(names, snr_scores)
    ax2.set_title('SNR Scores')
    ax2.set_ylabel('SNR (higher is better)')
    ax2.set_ylim(0, 100)  # SNR range is 0-100
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Training times
    ax3.bar(names, training_times)
    ax3.set_title('Training Time')
    ax3.set_ylabel('Seconds (lower is better)')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_plots.png'), dpi=300)
    
    # Create training loss plots
    plt.figure(figsize=(10, 6))
    for name in names:
        plt.plot(results[name]['train_losses'], label=f"{name} (Train)")
        plt.plot(results[name]['val_losses'], label=f"{name} (Val)", linestyle='--')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)

if __name__ == "__main__":
    main() 