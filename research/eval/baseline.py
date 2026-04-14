"""Baseline solution: simple linear fit."""
import numpy as np


def f(x):
    # Naive baseline: just return 0 (mean predictor)
    return np.zeros_like(x)
