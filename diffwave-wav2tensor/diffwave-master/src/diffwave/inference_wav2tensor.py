import os
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .wav2tensor_model import DiffWaveWav2Tensor
from .wav2tensor_params import params


def load_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DiffWaveWav2Tensor(checkpoint['params']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def generate_audio(model, audio_len, device, fast_sampling=True):
    """
    Generate audio using the model.
    """
    # Initialize noise
    audio = torch.randn(1, 1, audio_len, device=device)
    
    # Use inference noise schedule for faster sampling
    noise_schedule = params.inference_noise_schedule if fast_sampling else params.noise_schedule
    
    # Gradually denoise
    with torch.no_grad():
        for n in tqdm(range(len(noise_schedule) - 1, -1, -1)):
            # Get noise level
            noise_level = noise_schedule[n]
            
            # Predict and subtract noise
            predicted = model(audio, torch.tensor([n], device=device))
            audio = torch.sqrt(1.0 - noise_level) * audio - torch.sqrt(noise_level) * predicted
            
            # Clip values
            audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio.squeeze(0)


def save_audio(audio, output_path, sample_rate=22050):
    """
    Save generated audio to file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, audio.cpu(), sample_rate)
    print(f'Saved audio to: {output_path}')


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    checkpoint_path = 'checkpoints/wav2tensor/model_epoch_990.pt'  # Adjust path as needed
    model = load_checkpoint(checkpoint_path, device)
    
    # Generate audio
    audio_len = params.audio_len  # Default 5 seconds
    audio = generate_audio(model, audio_len, device, fast_sampling=True)
    
    # Save audio
    output_path = 'generated/wav2tensor_sample.wav'
    save_audio(audio, output_path)


if __name__ == '__main__':
    main() 