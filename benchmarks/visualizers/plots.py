"""
Visualization tools for benchmark results.

This module provides functions to visualize benchmark results,
including processing time, memory usage, and speedup metrics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_process_time(results: Dict, output_path: Optional[str] = None):
    """
    Plot processing time for each method.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the plot image (default: None, show plot)
    """
    methods = []
    times = []
    
    # Extract data
    for method in ['waveform', 'mel_spectrogram', 'wav2tensor', 'wav2tensor_lite']:
        if method in results:
            methods.append(method.replace('_', ' ').title())
            times.append(results[method]['avg_process_time'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, times, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.6f}s',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    # Set labels and title
    ax.set_ylabel('Processing Time (seconds)')
    ax.set_title('Audio Processing Time Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_memory_usage(results: Dict, output_path: Optional[str] = None):
    """
    Plot memory usage for each method.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the plot image (default: None, show plot)
    """
    methods = []
    memory = []
    
    # Extract data
    for method in ['waveform', 'mel_spectrogram', 'wav2tensor', 'wav2tensor_lite']:
        if method in results:
            methods.append(method.replace('_', ' ').title())
            memory.append(results[method]['avg_memory_kb'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, memory, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.1f} KB',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    # Set labels and title
    ax.set_ylabel('Memory Usage (KB)')
    ax.set_title('Memory Usage Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_speedup(results: Dict, output_path: Optional[str] = None):
    """
    Plot speedup factors between methods.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the plot image (default: None, show plot)
    """
    if 'speedup' not in results:
        print("No speedup data found in results")
        return
    
    # Extract data
    comparisons = []
    speedups = []
    
    for key, value in results['speedup'].items():
        if key.startswith('wav2tensor') and not np.isinf(value):
            # Format the label
            label = key.replace('_', ' ').replace('vs', 'vs.').title()
            comparisons.append(label)
            speedups.append(value)
    
    # Sort by speedup value
    sorted_indices = np.argsort(speedups)
    comparisons = [comparisons[i] for i in sorted_indices]
    speedups = [speedups[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(comparisons, speedups, color=['#9b59b6', '#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # Add values to the right of bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}x',
                ha='left', va='center', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Speedup Factor (higher is better)')
    ax.set_title('Relative Performance Comparison')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0, max(speedups) * 1.2)  # Add some space for labels
    
    # Set tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_summary(results: Dict, output_dir: Optional[str] = None):
    """
    Create and save all plots for a benchmark results file.
    
    Args:
        results: Dictionary containing benchmark results
        output_dir: Directory to save plots (default: None, use results dir)
    """
    # Determine output directory
    if output_dir is None:
        if 'output_dir' in results:
            output_dir = results['output_dir']
        else:
            output_dir = 'benchmarks/results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base filename
    base_name = os.path.splitext(os.path.basename(results.get('audio_file', 'benchmark')))
    time_str = results.get('timestamp', '')
    prefix = f"{output_dir}/{base_name[0]}_{time_str}"
    
    # Create and save plots
    plot_process_time(results, f"{prefix}_process_time.png")
    plot_memory_usage(results, f"{prefix}_memory_usage.png")
    plot_speedup(results, f"{prefix}_speedup.png")


def visualize_results_file(json_path: str):
    """
    Visualize results from a benchmark JSON file.
    
    Args:
        json_path: Path to benchmark results JSON file
    """
    # Load results file
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Add timestamp from filename if not in results
    if 'timestamp' not in results:
        filename = os.path.basename(json_path)
        if '_20' in filename:
            timestamp = filename.split('_')[1].split('.')[0]
            results['timestamp'] = timestamp
    
    # Create plots
    plot_summary(results)
    
    return f"Created visualizations for {json_path}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output_dir", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory if specified
    output_dir = args.output_dir
    
    # Create and save plots
    plot_summary(results, output_dir)
    
    print(f"Created visualizations for {args.results_file}")
    if output_dir:
        print(f"Saved to {output_dir}") 