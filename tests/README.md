# Wav2Tensor Test Suite

This directory contains tests for the Wav2Tensor project, organized into different categories.

## Test Organization

```
tests/
├── unit/               # Unit tests for individual components
│   ├── planes/         # Tests for individual planes
│   │   ├── test_spectral.py
│   │   ├── test_harmonic.py
│   │   └── ...
│   └── test_core.py    # Tests for core functionality
├── integration/        # Integration tests for components working together
├── performance/        # Performance and benchmarking tests
├── analysis/           # Analysis scripts (not automated tests)
│   └── harmonic_analyzer.py
└── conftest.py         # Shared fixtures for all tests
```

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test category:

```bash
# Run all unit tests
pytest tests/unit/

# Run all tests for planes
pytest tests/unit/planes/

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::TestWav2TensorCore::test_adaptive_frequency
```

## Test Fixtures

Common test fixtures are defined in `conftest.py` and include:

- `wav2tensor_default`: A default Wav2TensorCore instance
- `wav2tensor_with_adaptive_freq`: A Wav2TensorCore instance with adaptive frequency
- `wav2tensor_filterbank`: A Wav2TensorCore instance using the filterbank method
- `mono_waveform`: A simple mono test waveform (1 second sine tone)
- `stereo_waveform`: A simple stereo test waveform with phase difference
- `harmonic_waveform`: A test waveform with strong harmonic content
- `sample_audio_path`: Path to a sample audio file for testing
- `sample_audio_waveform`: Sample audio file loaded as waveform tensor

## Analysis Tools

The `analysis/` directory contains scripts for analyzing Wav2Tensor behavior, but they are not automated tests.

To run the harmonic analyzer:

```bash
# Basic usage
python -m tests.analysis.harmonic_analyzer path/to/audio.mp3

# With options
python -m tests.analysis.harmonic_analyzer path/to/audio.mp3 --duration 5 --start 10 --output-dir results --adaptive
``` 