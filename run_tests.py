#!/usr/bin/env python
"""
Run tests for the Wav2Tensor project with coverage.
"""

import os
import sys
import argparse
import subprocess


def run_tests(test_path=None, coverage=True, verbose=False):
    """Run tests with optional coverage."""
    try:
        # Install required test dependencies if not present
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        
        # Build the command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        
        # Add coverage if requested
        if coverage:
            cmd.extend(["--cov=wav2tensor", "--cov-report=term", "--cov-report=html"])
        
        # Add test path if specified, otherwise run all tests
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append("tests/")
        
        # Run the tests
        print(f"Running command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        if coverage:
            print("\nCoverage report generated in htmlcov/ directory")
            
        return 0
    
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return e.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for Wav2Tensor")
    parser.add_argument("test_path", nargs="?", help="Specific test path to run")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    sys.exit(run_tests(
        test_path=args.test_path,
        coverage=not args.no_coverage,
        verbose=args.verbose
    )) 