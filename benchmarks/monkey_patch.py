"""
Monkey patch for Wav2TensorCore to fix the _ensure_4d issue.

This module applies patches to Wav2TensorCore to make it compatible with benchmarking.
"""

import torch
from wav2tensor.core import Wav2TensorCore


def apply_patches():
    """Apply monkey patches to fix the _ensure_4d issue."""
    # Replace the _ensure_4d method with a properly typed version
    def ensure_4d_fixed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has shape [B, C, F, T] for consistent processing"""
        if len(tensor.shape) == 3:  # [B, F, T]
            return tensor.unsqueeze(1)
        return tensor
    
    # Replace the jit-scripted method with our fixed version
    Wav2TensorCore._ensure_4d = ensure_4d_fixed
    
    print("Applied monkey patch to fix Wav2TensorCore._ensure_4d")


# Apply patches when imported
apply_patches() 