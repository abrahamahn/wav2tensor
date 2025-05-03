import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pesq
import torchaudio.transforms as T
from tqdm import tqdm

# Add parent directory to path to import wav2tensor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from wav2tensor import Wav2TensorCore

# 1. Create Synthetic Dataset of Clean and Degraded Audio Pairs

class SyntheticAudioDataset(Dataset):
    """Dataset of synthetic clean + degraded audio pairs."""
    def __init__(self, num_samples=1000, sample_rate=22050, duration=1.0, 
                 snr_range=(5, 25), clip_range=(0.3, 0.9)):
        """
        Args:
            num_samples: Number of samples in the dataset
            sample_rate: Audio sample rate
            duration: Duration of each audio sample in seconds
            snr_range: Range of signal-to-noise ratios for degradation
            clip_range: Range of clipping thresholds for degradation
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples_per_audio = int(sample_rate * duration)
        self.snr_range = snr_range
        self.clip_range = clip_range
        
        # Initialize Wav2Tensor encoder
        self.wav2tensor = Wav2TensorCore(sample_rate=sample_rate)
    
    def __len__(self):
        return self.num_samples
    
    def add_noise(self, clean, snr_db):
        """Add noise at a specified SNR level."""
        clean_power = torch.mean(clean ** 2)
        noise = torch.randn_like(clean)
        noise_power = torch.mean(noise ** 2)
        
        # Scale noise to achieve desired SNR
        scale = torch.sqrt(clean_power / (10 ** (snr_db / 10) * noise_power))
        noise = scale * noise
        
        return clean + noise
    
    def apply_clipping(self, audio, threshold):
        """Apply hard clipping at the specified threshold."""
        return torch.clamp(audio, -threshold, threshold)
    
    def apply_reverb(self, audio, decay=0.5):
        """Apply simple artificial reverb."""
        # Create a simple impulse response
        ir_length = int(self.sample_rate * 0.3)  # 300ms impulse response
        ir = torch.zeros(ir_length)
        for i in range(ir_length):
            ir[i] = torch.exp(-decay * i / self.sample_rate)
        
        # Normalize IR
        ir = ir / torch.sum(ir)
        
        # Apply reverb using convolution
        reverb = torch.nn.functional.conv1d(
            audio.view(1, 1, -1), 
            ir.view(1, 1, -1), 
            padding=ir_length
        )
        
        return reverb.view_as(audio)
    
    def generate_synthetic_tone(self):
        """Generate a synthetic tone with harmonics."""
        # Generate a random fundamental frequency between 80-800 Hz
        f0 = torch.rand(1) * 720 + 80
        
        # Generate time axis
        t = torch.arange(0, self.duration, 1/self.sample_rate)
        
        # Generate a tone with harmonics
        signal = torch.zeros(self.num_samples_per_audio)
        num_harmonics = torch.randint(1, 8, (1,)).item()
        
        for i in range(1, num_harmonics + 1):
            # Amplitude decreases with harmonic number
            amplitude = 1.0 / i
            # Each harmonic has its own frequency
            signal += amplitude * torch.sin(2 * np.pi * i * f0 * t)
        
        # Add a random envelope to make it more natural
        envelope = torch.exp(-3 * torch.linspace(0, 1, self.num_samples_per_audio))
        envelope = torch.roll(envelope, torch.randint(0, self.num_samples_per_audio, (1,)).item())
        
        signal = signal * envelope
        
        # Normalize
        signal = signal / torch.max(torch.abs(signal))
        
        return signal
    
    def __getitem__(self, idx):
        """Generate a clean and degraded audio pair."""
        # Generate a random synthetic tone
        clean = self.generate_synthetic_tone()
        
        # Select random degradation type
        degradation_type = torch.randint(0, 3, (1,)).item()
        
        if degradation_type == 0:
            # Add noise
            snr = torch.rand(1) * (self.snr_range[1] - self.snr_range[0]) + self.snr_range[0]
            degraded = self.add_noise(clean, snr)
        elif degradation_type == 1:
            # Apply clipping
            threshold = torch.rand(1) * (self.clip_range[1] - self.clip_range[0]) + self.clip_range[0]
            degraded = self.apply_clipping(clean, threshold)
        else:
            # Apply reverb
            decay = torch.rand(1) * 0.8 + 0.1  # Decay between 0.1 and 0.9
            degraded = self.apply_reverb(clean, decay)
        
        # Normalize both signals to [-1, 1]
        clean = clean / (torch.max(torch.abs(clean)) + 1e-8)
        degraded = degraded / (torch.max(torch.abs(degraded)) + 1e-8)
        
        # Add batch and channel dimensions
        clean = clean.view(1, 1, -1)  # [1, 1, T]
        degraded = degraded.view(1, 1, -1)  # [1, 1, T]
        
        # Convert to Wav2Tensor representation
        clean_tensor, _ = self.wav2tensor(clean)
        degraded_tensor, _ = self.wav2tensor(degraded)
        
        return {
            'clean_waveform': clean.squeeze(0),
            'degraded_waveform': degraded.squeeze(0),
            'clean_tensor': clean_tensor.squeeze(0),
            'degraded_tensor': degraded_tensor.squeeze(0)
        }

# 2. Create a U-Net model for tensor enhancement

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Decoder
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with skip connections
        dec4 = self.up(enc4)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.dec1(dec2)
        
        return dec1

# 3. Create a basic spectrogram model for comparison

class SpectrogramUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            hop_length=256,
            power=2.0  # Magnitude squared
        )
        
        self.unet = UNet(in_channels=1, out_channels=1)
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=1024,
            hop_length=256
        )
    
    def forward(self, waveform):
        # Convert to spectrogram
        spec = torch.log1p(self.spec_transform(waveform))
        spec = spec.unsqueeze(1)  # Add channel dimension [B, 1, F, T]
        
        # Process with UNet
        enhanced_spec = self.unet(spec)
        
        # Convert back to linear scale
        enhanced_spec = torch.expm1(enhanced_spec.squeeze(1))
        
        # ISTFT (approximate phase reconstruction)
        enhanced_waveform = self.istft(enhanced_spec.sqrt())
        
        return enhanced_waveform

# 4. Evaluation metrics

def calculate_pesq(clean, enhanced, sr=22050):
    """Calculate PESQ score between clean and enhanced audio."""
    try:
        # Convert to numpy arrays
        clean_np = clean.cpu().numpy()
        enhanced_np = enhanced.cpu().numpy()
        
        # Normalize to appropriate range for PESQ
        clean_np = clean_np / np.max(np.abs(clean_np))
        enhanced_np = enhanced_np / np.max(np.abs(enhanced_np))
        
        # Calculate PESQ
        score = pesq.pesq(sr, clean_np, enhanced_np, 'wb')
        return score
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return 0.0

def calculate_stoi(clean, enhanced, sr=22050):
    """Calculate STOI score between clean and enhanced audio."""
    try:
        # Convert to numpy arrays
        clean_np = clean.cpu().numpy()
        enhanced_np = enhanced.cpu().numpy()
        
        # Normalize to appropriate range
        clean_np = clean_np / np.max(np.abs(clean_np))
        enhanced_np = enhanced_np / np.max(np.abs(enhanced_np))
        
        # Calculate STOI (you'll need to install pystoi for this)
        from pystoi import stoi
        score = stoi(clean_np, enhanced_np, sr, extended=False)
        return score
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        return 0.0

# 5. Training function

def train_model(model, dataloader, device, num_epochs=10, learning_rate=0.001):
    """Train a model on the dataloader."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if isinstance(model, UNet):
                # Tensor model
                inputs = batch['degraded_tensor'].to(device)
                targets = batch['clean_tensor'].to(device)
            else:
                # Spectrogram model
                inputs = batch['degraded_waveform'].to(device)
                targets = batch['clean_waveform'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    return model

# 6. Evaluation function

def evaluate_model(model, wav2tensor, test_dataloader, device, model_type='tensor'):
    """Evaluate model performance on test data."""
    model.eval()
    pesq_scores = []
    stoi_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            clean_waveform = batch['clean_waveform'].to(device)
            degraded_waveform = batch['degraded_waveform'].to(device)
            
            if model_type == 'tensor':
                # Process with tensor model
                degraded_tensor = batch['degraded_tensor'].to(device)
                enhanced_tensor = model(degraded_tensor)
                
                # Convert back to waveform (this is a simplified reconstruction)
                # In a real implementation, you would need a proper tensor-to-waveform converter
                enhanced_waveform = reconstruct_waveform(enhanced_tensor, wav2tensor)
            else:
                # Process with spectrogram model
                enhanced_waveform = model(degraded_waveform)
            
            # Calculate metrics
            for i in range(clean_waveform.size(0)):
                pesq_score = calculate_pesq(
                    clean_waveform[i].squeeze().cpu(), 
                    enhanced_waveform[i].squeeze().cpu()
                )
                stoi_score = calculate_stoi(
                    clean_waveform[i].squeeze().cpu(), 
                    enhanced_waveform[i].squeeze().cpu()
                )
                
                pesq_scores.append(pesq_score)
                stoi_scores.append(stoi_score)
    
    avg_pesq = sum(pesq_scores) / len(pesq_scores)
    avg_stoi = sum(stoi_scores) / len(stoi_scores)
    
    return {
        'PESQ': avg_pesq,
        'STOI': avg_stoi
    }

# Helper function for waveform reconstruction
def reconstruct_waveform(tensor, wav2tensor):
    """
    A placeholder for tensor-to-waveform reconstruction.
    In a real implementation, you would need a proper decoder network.
    """
    # This is a very simplified reconstruction for demonstration
    # Real implementation would require a learned decoder or GLA
    B, D, F, T = tensor.shape
    
    # Extract real and imaginary parts of spectral plane
    spec_real = tensor[:, 0]
    spec_imag = tensor[:, 1]
    
    # Reconstruct complex spectrogram
    complex_spec = torch.complex(spec_real, spec_imag)
    
    # Apply inverse STFT (this is a very simplified approach)
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=wav2tensor.n_fft,
        hop_length=wav2tensor.hop_length
    )
    
    # Reconstruct waveform
    waveform = istft(complex_spec)
    
    return waveform

# 7. Main function

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SyntheticAudioDataset(num_samples=1000)
    test_dataset = SyntheticAudioDataset(num_samples=100)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Get tensor dimensions from a sample
    sample = next(iter(train_dataloader))
    in_channels = sample['degraded_tensor'].size(1)
    out_channels = sample['clean_tensor'].size(1)
    
    # Create models
    tensor_model = UNet(in_channels=in_channels, out_channels=out_channels)
    spectrogram_model = SpectrogramUNet()
    
    # Train models
    print("Training Wav2Tensor model...")
    tensor_model = train_model(tensor_model, train_dataloader, device, num_epochs=5)
    
    print("\nTraining Spectrogram model...")
    spectrogram_model = train_model(spectrogram_model, train_dataloader, device, num_epochs=5)
    
    # Evaluate models
    wav2tensor = Wav2TensorCore()
    
    print("\nEvaluating Wav2Tensor model...")
    tensor_metrics = evaluate_model(tensor_model, wav2tensor, test_dataloader, device, model_type='tensor')
    
    print("\nEvaluating Spectrogram model...")
    spec_metrics = evaluate_model(spectrogram_model, wav2tensor, test_dataloader, device, model_type='spectrogram')
    
    # Print results
    print("\n==== Results ====")
    print(f"Wav2Tensor model: PESQ = {tensor_metrics['PESQ']:.3f}, STOI = {tensor_metrics['STOI']:.3f}")
    print(f"Spectrogram model: PESQ = {spec_metrics['PESQ']:.3f}, STOI = {spec_metrics['STOI']:.3f}")
    
    # Plot a sample result
    with torch.no_grad():
        sample = next(iter(test_dataloader))
        clean_waveform = sample['clean_waveform'][0].squeeze().to(device)
        degraded_waveform = sample['degraded_waveform'][0].squeeze().to(device)
        
        # Process with tensor model
        degraded_tensor = sample['degraded_tensor'][0].unsqueeze(0).to(device)
        enhanced_tensor = tensor_model(degraded_tensor)
        enhanced_waveform_tensor = reconstruct_waveform(enhanced_tensor, wav2tensor)
        
        # Process with spectrogram model
        degraded_spec = degraded_waveform.unsqueeze(0).to(device)
        enhanced_waveform_spec = spectrogram_model(degraded_spec)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(4, 1, 1)
        plt.title("Clean Waveform")
        plt.plot(clean_waveform.cpu().numpy())
        
        plt.subplot(4, 1, 2)
        plt.title("Degraded Waveform")
        plt.plot(degraded_waveform.cpu().numpy())
        
        plt.subplot(4, 1, 3)
        plt.title("Enhanced Waveform (Wav2Tensor model)")
        plt.plot(enhanced_waveform_tensor.squeeze().cpu().numpy())
        
        plt.subplot(4, 1, 4)
        plt.title("Enhanced Waveform (Spectrogram model)")
        plt.plot(enhanced_waveform_spec.squeeze().cpu().numpy())
        
        plt.tight_layout()
        plt.savefig("waveform_comparison.png")
        plt.close()
        
        # Plot spectrograms
        plt.figure(figsize=(12, 10))
        
        # Create spectrogram transform
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=256, power=2.0
        )
        
        # Compute spectrograms
        clean_spec = spec_transform(clean_waveform.unsqueeze(0))
        degraded_spec = spec_transform(degraded_waveform.unsqueeze(0))
        enhanced_spec_tensor = spec_transform(enhanced_waveform_tensor.squeeze().unsqueeze(0))
        enhanced_spec_spec = spec_transform(enhanced_waveform_spec.squeeze().unsqueeze(0))
        
        # Plot in dB scale
        plt.subplot(2, 2, 1)
        plt.title("Clean Spectrogram")
        plt.imshow(torch.log10(clean_spec.squeeze() + 1e-8).cpu().numpy(), 
                  aspect='auto', origin='lower')
        
        plt.subplot(2, 2, 2)
        plt.title("Degraded Spectrogram")
        plt.imshow(torch.log10(degraded_spec.squeeze() + 1e-8).cpu().numpy(),
                  aspect='auto', origin='lower')
        
        plt.subplot(2, 2, 3)
        plt.title("Enhanced Spectrogram (Wav2Tensor model)")
        plt.imshow(torch.log10(enhanced_spec_tensor.squeeze() + 1e-8).cpu().numpy(),
                  aspect='auto', origin='lower')
        
        plt.subplot(2, 2, 4)
        plt.title("Enhanced Spectrogram (Spectrogram model)")
        plt.imshow(torch.log10(enhanced_spec_spec.squeeze() + 1e-8).cpu().numpy(),
                  aspect='auto', origin='lower')
        
        plt.tight_layout()
        plt.savefig("spectrogram_comparison.png")
        plt.close()

if __name__ == "__main__":
    main() 