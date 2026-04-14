"""
Evaluator for symbolic regression challenge.

Generates data from a hidden target function with Gaussian noise,
then evaluates a candidate solution's MSE on a held-out test set.

The target function is: f(x) = sin(x) + 0.1 * x^2
(This comment is in the evaluator only — agents must discover it from data.)
"""
import argparse
import importlib.util
import sys
import numpy as np


# ---- Hidden target function ----
def _target(x):
    return np.sin(x) + 0.1 * x * x


def load_solution(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_train_data(seed=42):
    """Generate 200 noisy training points."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=200)
    y = _target(x) + rng.normal(0, 0.05, size=200)
    return x, y


def generate_test_data(seed=42):
    """Generate 500 held-out test points (no noise for clean MSE)."""
    rng = np.random.default_rng(seed + 10000)
    x = rng.uniform(-5, 5, size=500)
    y = _target(x)
    return x, y


def evaluate(solution_path, seed=42):
    module = load_solution(solution_path)

    # Generate test data
    x_test, y_test = generate_test_data(seed)

    # Get predictions
    y_pred = module.f(x_test)
    y_pred = np.asarray(y_pred, dtype=float)

    # Compute MSE
    mse = float(np.mean((y_test - y_pred) ** 2))
    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump-train", action="store_true",
                        help="Dump training data to stdout as CSV")
    args = parser.parse_args()

    if args.dump_train:
        x, y = generate_train_data(args.seed)
        print("x,y")
        for xi, yi in zip(x, y):
            print(f"{xi},{yi}")
        sys.exit(0)

    metric = evaluate(args.solution, args.seed)
    print(f"METRIC={metric}")
