#!/usr/bin/env python3
"""
Simple script to check if GPU is available in container
Used for Docker healthchecks
"""
import os
import sys

try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        sys.exit(0)  # Success
    else:
        print("No GPU found!")
        sys.exit(1)  # Failure
except Exception as e:
    print(f"Error checking GPU: {e}")
    sys.exit(1)  # Failure 