"""
Progressive Training Example with Wav2TensorLite

This example demonstrates how to implement a progressive training approach with Wav2TensorLite,
gradually adding planes during training to improve convergence and performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wav2tensor.core_lite import Wav2TensorLite

# Simple UNet model for demonstration
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Save original input size for potential padding
        orig_h, orig_w = x.shape[2], x.shape[3]
        
        # Ensure dimensions are divisible by 4 (for 2 pooling operations)
        h, w = ((orig_h + 3) // 4) * 4, ((orig_w + 3) // 4) * 4
        if h != orig_h or w != orig_w:
            # Pad to make dimensions divisible by 4
            pad_h = h - orig_h
            pad_w = w - orig_w
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder with skip connections
        d1 = self.upconv1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        # Output
        out = self.output(d2)
        
        # Crop back to original size if needed
        if h != orig_h or w != orig_w:
            out = out[:, :, :orig_h, :orig_w]
            
        return out

def generate_synthetic_audio(duration=2.0, sample_rate=16000):
    """Generate a synthetic audio signal with noise for demonstration"""
    # Generate sine waves at different frequencies
    t = torch.arange(0, duration, 1.0/sample_rate)
    
    # Create a clean signal with multiple frequency components
    clean = (
        0.5 * torch.sin(2 * np.pi * 440 * t) +  # A4
        0.3 * torch.sin(2 * np.pi * 880 * t) +  # A5 (2nd harmonic)
        0.2 * torch.sin(2 * np.pi * 1320 * t)   # E6 (3rd harmonic)
    )
    
    # Add noise to create the noisy version
    noise = torch.randn_like(clean) * 0.1
    noisy = clean + noise
    
    # Normalize
    clean = clean / torch.max(torch.abs(clean))
    noisy = noisy / torch.max(torch.abs(noisy))
    
    # Create stereo versions
    clean_stereo = torch.stack([clean, clean])
    noisy_stereo = torch.stack([noisy, noisy])
    
    return clean_stereo, noisy_stereo

def visualize_training_progression(losses, plane_configs):
    """Visualize the training progression with different plane configurations"""
    plt.figure(figsize=(12, 6))
    
    # Plot the loss curve
    plt.subplot(1, 2, 1)
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'b-')
    
    # Mark where plane configurations changed
    change_points = [0]
    for i in range(1, len(plane_configs)):
        if plane_configs[i] != plane_configs[i-1]:
            change_points.append(i)
    
    # Add markers for plane configuration changes
    for cp in change_points:
        if cp > 0:  # Skip the first point
            plt.axvline(x=cp, color='r', linestyle='--', alpha=0.7)
    
    plt.title('Training Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Add plane configuration annotations
    unique_configs = []
    for i, cp in enumerate(change_points):
        end = change_points[i+1] if i+1 < len(change_points) else len(losses)
        config = plane_configs[cp]
        if config not in [c[0] for c in unique_configs]:
            unique_configs.append((config, cp, end))
    
    for config, start, end in unique_configs:
        mid = (start + end) // 2
        plt.text(mid, max(losses) * 0.9, f"{config}", 
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot the loss improvement for each configuration
    plt.subplot(1, 2, 2)
    for i, (config, start, end) in enumerate(unique_configs):
        if end > start + 1:  # Only if we have at least 2 points
            # Calculate improvement
            start_loss = losses[start]
            end_loss = losses[end-1]
            improvement = (start_loss - end_loss) / start_loss * 100
            
            plt.bar(i, improvement, width=0.6)
            plt.text(i, improvement + 1, f"{improvement:.1f}%", 
                    horizontalalignment='center')
    
    plt.title('Loss Improvement per Configuration')
    plt.xlabel('Configuration')
    plt.xticks(range(len(unique_configs)), [c[0] for c in unique_configs], rotation=45)
    plt.ylabel('Improvement %')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('progressive_training_results.png', dpi=200)
    print("Visualization saved to progressive_training_results.png")

def progressive_training_demo():
    """Demonstrate progressive training with Wav2TensorLite"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    sample_rate = 16000
    n_fft = 512
    hop_length = 128
    batch_size = 4
    num_epochs_per_config = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset_size = 20
    clean_audios = []
    noisy_audios = []
    
    for i in range(dataset_size):
        clean, noisy = generate_synthetic_audio(duration=2.0, sample_rate=sample_rate)
        clean_audios.append(clean)
        noisy_audios.append(noisy)
    
    # Define progressive training configurations
    progressive_configs = [
        {
            "name": "Spectral Only",
            "planes": ["spectral"],
            "fusion": "concat",
            "bit_depth": 16,
            "adaptive_freq": True
        },
        {
            "name": "Spectral+Harmonic",
            "planes": ["spectral", "harmonic"],
            "fusion": "concat",
            "bit_depth": 16,
            "adaptive_freq": True
        },
        {
            "name": "All Planes",
            "planes": ["spectral", "harmonic", "spatial", "psychoacoustic"],
            "fusion": "learned",
            "bit_depth": 16,
            "adaptive_freq": True
        }
    ]
    
    # Arrays to track training progression
    all_losses = []
    plane_config_names = []
    
    # Train with each configuration
    for config_idx, config in enumerate(progressive_configs):
        print(f"\nTraining with configuration {config_idx+1}/{len(progressive_configs)}: {config['name']}")
        print(f"Planes: {config['planes']}, Fusion: {config['fusion']}")
        
        # Create Wav2TensorLite encoder for this configuration
        encoder = Wav2TensorLite(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            bit_depth=config['bit_depth'],
            use_adaptive_freq=config['adaptive_freq'],
            harmonic_method='hps',
            include_planes=config['planes'],
            fusion_method=config['fusion']
        ).to(device)
        
        # Process the entire dataset with the current encoder
        clean_tensors = []
        noisy_tensors = []
        
        print("Processing dataset with current configuration...")
        for i in range(dataset_size):
            with torch.no_grad():
                # Convert clean audio
                clean_tensor, _ = encoder(clean_audios[i].unsqueeze(0).to(device))
                clean_tensors.append(clean_tensor.cpu())
                
                # Convert noisy audio
                noisy_tensor, _ = encoder(noisy_audios[i].unsqueeze(0).to(device))
                noisy_tensors.append(noisy_tensor.cpu())
        
        # Determine input/output channels based on encoder output
        in_channels = noisy_tensors[0].shape[1]
        out_channels = clean_tensors[0].shape[1]
        
        # Create model (new for each configuration to simulate starting fresh)
        model = SimpleUNet(in_channels, out_channels).to(device)
        
        # If not the first configuration, we could load weights from previous model
        # and adapt them (this would be more complex in a real scenario)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop for this configuration
        for epoch in range(num_epochs_per_config):
            model.train()
            epoch_loss = 0
            
            # Create batches
            indices = torch.randperm(dataset_size)
            num_batches = (dataset_size + batch_size - 1) // batch_size
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Create batch tensors
                batch_noisy = torch.cat([noisy_tensors[i] for i in batch_indices]).to(device)
                batch_clean = torch.cat([clean_tensors[i] for i in batch_indices]).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_noisy)
                loss = criterion(outputs, batch_clean)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
            
            # Calculate average epoch loss
            epoch_loss /= dataset_size
            all_losses.append(epoch_loss)
            plane_config_names.append(config['name'])
            
            print(f"Epoch {epoch+1}/{num_epochs_per_config}, Loss: {epoch_loss:.6f}")
    
    # Visualize training progression
    visualize_training_progression(all_losses, plane_config_names)

if __name__ == "__main__":
    progressive_training_demo() 