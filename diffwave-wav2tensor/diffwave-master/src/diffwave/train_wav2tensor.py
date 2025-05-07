import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .wav2tensor_model import DiffWaveWav2Tensor
from .wav2tensor_params import params
from .dataset import AudioDataset


def compute_diffusion_loss(model, audio, diffusion_step):
    """
    Compute the diffusion loss for a batch of audio.
    """
    # Add noise according to diffusion step
    noise = torch.randn_like(audio)
    noise_level = params.noise_schedule[diffusion_step]
    noisy_audio = torch.sqrt(1.0 - noise_level) * audio + torch.sqrt(noise_level) * noise
    
    # Predict noise
    predicted = model(noisy_audio, diffusion_step)
    
    # Compute loss
    loss = F.mse_loss(predicted, noise)
    return loss


def train(model, train_loader, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as progress:
        for batch_idx, audio in enumerate(progress):
            audio = audio.to(device)
            optimizer.zero_grad()
            
            # Randomly sample diffusion step
            diffusion_step = torch.randint(0, len(params.noise_schedule), (audio.shape[0],), device=device)
            
            # Compute loss
            loss = compute_diffusion_loss(model, audio, diffusion_step)
            
            # Backprop
            loss.backward()
            if params.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress.set_postfix(loss=f'{avg_loss:.4f}')
    
    return total_loss / len(train_loader)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """
    Save model checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'params': params,
    }, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = DiffWaveWav2Tensor(params).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    # Create dataset and dataloader
    dataset = AudioDataset(params)
    train_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Training loop
    num_epochs = 1000  # Adjust as needed
    checkpoint_dir = 'checkpoints/wav2tensor'
    
    for epoch in range(num_epochs):
        # Train for one epoch
        loss = train(model, train_loader, optimizer, device, epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir)
        
        print(f'Epoch {epoch} - Average loss: {loss:.4f}')


if __name__ == '__main__':
    main() 