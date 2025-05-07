import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=22050,
    n_fft=1024,
    hop_samples=256,
    
    # Wav2Tensor params
    use_adaptive_freq=False,
    harmonic_method='hps',
    include_planes=['spectral', 'harmonic', 'spatial', 'psychoacoustic'],

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # Audio length for generation
    audio_len=22050*5,  # 5 seconds at 22.05kHz
) 