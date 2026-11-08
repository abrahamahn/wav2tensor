"""
Configuration loader for benchmarks.

This module provides functions to load and save benchmark configurations
from YAML files, with validation and default values.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import yaml

from wav2tensor.frequency import VALID_PLANES

DEFAULT_CONFIG: dict[str, Any] = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "harmonic_method": "hps",
    "use_adaptive_freq": False,
    "target_freq_bins": 256,
    "include_planes": list(VALID_PLANES),
    "n_runs": 5,
    "segment_duration": None,
    "start_time": 0.0,
    "output_dir": "benchmarks/results",
}


def parse_planes_arg(planes_arg: str) -> list[str]:
    """Parse CLI `--planes` specification into an explicit plane list."""
    if planes_arg == "default":
        return ["spectral", "harmonic", "spatial"]
    if planes_arg == "all":
        return list(VALID_PLANES)
    if planes_arg == "minimal":
        return ["spectral"]
    return [plane.strip() for plane in planes_arg.split(",") if plane.strip()]


def load_config(config_path: str) -> dict[str, Any]:
    """Load a benchmark configuration from a YAML file with defaults applied."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = default_value

    _validate_config(config)
    return config


def save_config(config: dict[str, Any], config_path: str) -> None:
    """Save a benchmark configuration to a YAML file."""
    _validate_config(config)
    parent = os.path.dirname(config_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    serializable_config = _convert_to_serializable(config)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(serializable_config, f, default_flow_style=False)


def _validate_config(config: dict[str, Any]) -> None:
    if "sample_rate" not in config:
        raise ValueError("Configuration must include 'sample_rate'")

    if "harmonic_method" in config and config["harmonic_method"] not in ("hps", "filterbank"):
        raise ValueError("harmonic_method must be 'hps' or 'filterbank'")

    if "include_planes" in config:
        for plane in config["include_planes"]:
            if plane not in VALID_PLANES:
                raise ValueError(f"Invalid plane: {plane}. Must be one of {list(VALID_PLANES)}")


def _convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_to_serializable(v) for v in obj)
    if isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def create_default_config(output_path: str, audio_file: str | None = None) -> None:
    """Create a default configuration file."""
    config = DEFAULT_CONFIG.copy()
    if audio_file is not None:
        config["audio_file"] = audio_file
    save_config(config, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a default benchmark configuration file")
    parser.add_argument("output_path", help="Path to save the default configuration file")
    parser.add_argument("--audio", help="Audio file to include in the configuration")
    args = parser.parse_args()
    create_default_config(args.output_path, args.audio)
