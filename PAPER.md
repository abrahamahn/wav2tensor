# Wav2Tensor: Structured Multi-Plane Audio Representation for Enhanced Generative Modeling  
**Abraham Joongwhan Ahn**  
Yonsei University, Seoul, South Korea  
satmorningrain@gmail.com  

## Abstract  
Modern audio processing models rely on suboptimal representations like spectrograms or raw waveforms, which often discard critical phase, harmonic, and spatial information.  
We propose **Wav2Tensor**, a novel structured multi-plane audio representation that explicitly encodes spectral, phase, harmonic, spatial, and psychoacoustic attributes into a unified tensor.  
By preserving physically and perceptually meaningful features, Wav2Tensor enables computationally efficient, high-fidelity audio reconstruction and enhancement across generative tasks.  
Benchmarks demonstrate that downstream models conditioned on Wav2Tensor achieve superior objective metrics (PESQ: $$3.72$$, STOI: $$0.91$$) compared to spectrogram-based baselines (PESQ: $$3.12$$),  
while reducing training convergence time by $$37\%$$.

**Keywords**: Audio Representation Learning, Tensor Encoding, Phase Preservation, Computational Efficiency

## 1. Introduction

### 1.1 Motivation  
Traditional audio representations suffer from critical limitations:  
- **Spectrograms**: Lose phase information, crucial for spatial and perceptual coherence.  
- **Raw Waveforms**: High dimensionality and lack of structure impede efficient learning.  
- **Existing Hybrids**: Fail to unify harmonic, spatial, and psychoacoustic cues.  

Wav2Tensor addresses these gaps by introducing a structured multi-plane tensor that preserves phase, harmonic structure, spatial geometry, and perceptual attributes while remaining computationally tractable.

### 1.2 Contributions  
**Multi-Plane Audio Tensor**: A structured representation with explicit encoding of:  
- Spectral magnitude and phase ($$T_{spec}$$)  
- Harmonic structure ($$T_{harm}$$)  
- Spatial cues ($$T_{spat}$$)  
- Psychoacoustic features ($$T_{psy}$$)  

**Computational Efficiency**: $$2.3\times$$ faster training convergence vs. waveform models.  
**Plug-and-Play Compatibility**: Seamless integration with GANs, diffusion models, and transformers.

## 2. Wav2Tensor: Structured Multi-Plane Representation

### 2.1 Tensor Construction  
Given an input waveform $$y \in \mathbb{R}^T$$, Wav2Tensor encodes it into a tensor $$T \in \mathbb{C}^{F \times T' \times D}$$ with four interpretable planes:

**1. Spectral Plane ($$d=1$$)**  
Combines magnitude and phase via the STFT:  
$$T_{spec}(f,t) = \text{STFT}(y)_{f,t} \in \mathbb{C}$$

**2. Harmonic Plane ($$d=2$$)**  
Captures pitch and overtones using Harmonic Product Spectrum (HPS):  
$$T_{harm}(f,t) = \prod_{k=1}^{K} |\text{STFT}(y)_{f/k,t}|$$

> *Implementation Note*: For numerical stability, the actual implementation computes this in the log-domain:  
> $$T_{harm}(f,t) = \exp\left(\sum_{k=1}^{K} \log(1 + |\text{STFT}(y)_{f/k,t}|)\right) - 1$$

**3. Spatial Plane ($$d=3$$)**  
Encodes stereo spatiality via interaural phase difference (IPD) and energy panning:  
$$T_{spat}(f,t) = \left\{ \text{IPD}(y_L, y_R)_{f,t}, \frac{|y_L|^2 - |y_R|^2}{|y_L|^2 + |y_R|^2} \right\}$$

**4. Psychoacoustic Plane ($$d=4$$)**  
Estimates perceptual masking thresholds (MPEG-inspired):  
$$T_{psy}(f,t) = M(y)_{f,t}$$

> *Implementation Note*: Our implementation models frequency-dependent masking thresholds using Bark scale conversion and asymmetric spreading functions:  
> $$T_{psy}(f,t) = \max(0, M_{spread}(y)_{f,t} - \tau)$$  
> where $$M_{spread}$$ applies a psychoacoustic spreading function and $$\tau$$ represents a masking threshold offset.

### 2.2 Computational Efficiency  
Wav2Tensor reduces model complexity by providing structured priors, avoiding redundant feature rediscovery:  

| Representation | Dimensions          | Training Speed (steps/sec) | GPU Memory (GB) |
|----------------|---------------------|-----------------------------|------------------|
| Raw Waveform   | $$T = 16,000$$      | 112                         | 4.2              |
| Mel-Spectrogram| $$F \times T'$$    | 203                         | 2.1              |
| Wav2Tensor     | $$F \times T' \times D$$ | 298                     | 2.8              |

Tested on NVIDIA RTX 3090 with $$T = 1$$s audio at 16 kHz.

## 3. Experiments

### 3.1 Downstream Task Performance  
We integrated Wav2Tensor into three generative models:

| Model     | Representation   | PESQ $$\uparrow$$ | STOI $$\uparrow$$ | SDR (dB) $$\uparrow$$ |
|-----------|------------------|--------------------|---------------------|------------------------|
| HiFi-GAN  | Mel-Spectrogram  | 3.12               | 0.82                | 8.1                    |
| HiFi-GAN  | Wav2Tensor       | 3.72               | 0.91                | 11.3                   |
| DiffWave  | Raw Waveform     | 3.54               | 0.89                | 10.7                   |
| DiffWave  | Wav2Tensor       | 3.81               | 0.93                | 12.1                   |

### 3.2 Ablation Study  
Removing individual planes degrades performance:

| Removed Plane       | PESQ $$\downarrow$$ | STOI $$\downarrow$$ |
|---------------------|----------------------|-----------------------|
| Harmonic ($$d=2$$)  | 3.21                 | 0.84                  |
| Spatial ($$d=3$$)   | 3.55                 | 0.88                  |
| Psychoacoustic ($$d=4$$) | 3.63           | 0.89                  |

## 4. Conclusion  
Wav2Tensor provides a structured, efficient audio representation that bridges the gap between signal processing and deep learning.  
By explicitly encoding phase, harmonic, spatial, and perceptual features, it enables higher-fidelity generative modeling with reduced computational overhead.  
Future work will extend the tensor to 3D spatial audio and edge-device deployment.
