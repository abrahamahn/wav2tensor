"""
Configuration management for benchmarks.

This module provides tools for loading, validating, and applying
benchmark configurations from YAML files.
"""

from benchmarks.configs.config_loader import load_config, save_config

__all__ = ['load_config', 'save_config'] 