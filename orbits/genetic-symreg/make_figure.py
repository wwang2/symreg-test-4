"""Generate visualization of the discovered function vs training data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# Load training data
xs, ys = [], []
with open("/tmp/train_data.csv") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))
xs, ys = np.array(xs), np.array(ys)

# Discovered function
x_smooth = np.linspace(-5, 5, 500)
y_pred = np.sin(x_smooth) + 0.1 * x_smooth ** 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(xs, ys, s=10, alpha=0.5, label="Training data (noisy)")
ax.plot(x_smooth, y_pred, "r-", lw=2, label=r"$f(x) = \sin(x) + 0.1\,x^2$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Genetic Symbolic Regression: Discovered Function")
ax.legend()
fig.tight_layout()
fig.savefig("orbits/genetic-symreg/figures/fit.png", dpi=150)
print("Figure saved.")
