"""
Configuration loader for benchmarks.

This module provides functions to load and save benchmark configurations 
from YAML files, with validation and default values.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union


# Default configuration values
DEFAULT_CONFIG = {
    'sample_rate': 22050,
    'n_fft': 1024,
    'hop_length': 256,
    'n_mels': 80,
    'harmonic_method': 'hps',
    'use_adaptive_freq': False,
    'target_freq_bins': 256,
    'include_planes': ['spectral', 'harmonic', 'spatial', 'psychoacoustic'],
    'n_runs': 5,
    'segment_duration': None,
    'start_time': 0.0,
    'output_dir': 'benchmarks/results'
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a benchmark configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing the configuration with defaults applied
        
    Raises:
        FileNotFoundError: If the config file does not exist
        yaml.YAMLError: If the config file is not valid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply defaults for missing values
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value
    
    # Validate configuration
    _validate_config(config)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save a benchmark configuration to a YAML file.
    
    Args:
        config: Dict containing the configuration
        config_path: Path to save the YAML configuration file
        
    Raises:
        ValueError: If the config is not valid
    """
    # Validate configuration
    _validate_config(config)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Convert any numpy/torch types to Python native types
    config = _convert_to_serializable(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def _validate_config(config: Dict[str, Any]):
    """
    Validate benchmark configuration.
    
    Args:
        config: Dict containing the configuration
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Required fields
    if 'sample_rate' not in config:
        raise ValueError("Configuration must include 'sample_rate'")
    
    # Valid harmonic methods
    if 'harmonic_method' in config and config['harmonic_method'] not in ['hps', 'filterbank']:
        raise ValueError("harmonic_method must be 'hps' or 'filterbank'")
    
    # Valid planes
    valid_planes = ['spectral', 'harmonic', 'spatial', 'psychoacoustic']
    if 'include_planes' in config:
        for plane in config['include_planes']:
            if plane not in valid_planes:
                raise ValueError(f"Invalid plane: {plane}. Must be one of {valid_planes}")


def _convert_to_serializable(obj: Any) -> Any:
    """
    Convert any non-serializable objects to serializable ones.
    
    Args:
        obj: The object to convert
        
    Returns:
        A serializable version of the object
    """
    import numpy as np
    import torch
    
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(v) for v in obj)
    elif isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    else:
        return str(obj)


def create_default_config(output_path: str, audio_file: str = None):
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save the default configuration
        audio_file: Optional audio file path to include in the configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    if audio_file is not None:
        config['audio_file'] = audio_file
    
    save_config(config, output_path)
    print(f"Default configuration saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a default benchmark configuration file")
    parser.add_argument("output_path", help="Path to save the default configuration file")
    parser.add_argument("--audio", help="Audio file to include in the configuration")
    
    args = parser.parse_args()
    
    create_default_config(args.output_path, args.audio) 