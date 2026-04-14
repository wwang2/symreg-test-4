"""Solution: f(x) = sin(x) + 0.1 * x^2

Discovered via polynomial fitting with trig basis expansion.
Degree-2 polynomial + sin/cos terms yielded coefficients:
  sin(x): ~1.007, x^2: ~0.100, all others near zero.
"""
import numpy as np


def f(x):
    return np.sin(x) + 0.1 * x ** 2
