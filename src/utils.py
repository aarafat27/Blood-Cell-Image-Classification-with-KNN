# File: C:\Users\Arafat\Desktop\IMA Project\src\utils.py

"""
In this file I want to put small helper functions that don't belong to
a specific topic like features or preprocessing. Things like setting
random seeds, timing code, printing shapes, etc.
"""

import os
import time
import random
import numpy as np


def set_random_seed(seed_value: int = 42):
    """
    In this function I want to fix all random seeds,
    so that my experiments are reproducible.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import torch  # just in case, if later I use torch

        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass
    print(f"[utils] Random seed set to {seed_value}")


class SimpleTimer:
    """
    In this class I want a very simple timer,
    so I can measure how long some parts of my code take.
    """

    def __init__(self, message: str = "Elapsed time"):
        self.message = message
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[timer] Starting: {self.message}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"[timer] {self.message}: {duration:.2f} seconds")


def print_array_info(temp_array, name: str = "array"):
    """
    In this function I want to print basic info about a numpy array,
    like shape, dtype, min, max, mean. This is useful for debugging.
    """
    print(f"--- {name} info ---")
    print(f"Shape: {temp_array.shape}")
    print(f"Dtype: {temp_array.dtype}")
    print(f"Min: {temp_array.min()}, Max: {temp_array.max()}, Mean: {temp_array.mean():.4f}")
    print("-------------------")
