# Symbolic Regression from Noisy Data

## Problem Statement
Given 200 noisy (x, y) data points, find a symbolic expression f(x) that best fits the data. The data is generated from an unknown target function with additive Gaussian noise (sigma=0.05). The input x ranges from -5 to 5. Agents must discover the underlying function from the data alone.

## Solution Interface
Solution must be a Python file (`solution.py`) with a function `f(x)` that takes a numpy array and returns predictions as a numpy array.

## Success Metric
Mean Squared Error (minimize) on a held-out test set of 500 points. Target: MSE < 0.01.
