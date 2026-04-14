"""
Systematic curve fitting with trigonometric + polynomial models.
Uses scipy curve_fit and BIC for model selection.
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

# Load training data
data_path = "/tmp/train_data.csv"
xs, ys = [], []
with open(data_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
x = np.array(xs)
y = np.array(ys)

n = len(x)

# Define candidate models
def model_1(x, a, b, c, d):
    """a*sin(x) + b*x^2 + c*x + d"""
    return a * np.sin(x) + b * x**2 + c * x + d

def model_2(x, a, b, c, d, e):
    """a*sin(b*x) + c*x^2 + d*x + e"""
    return a * np.sin(b * x) + c * x**2 + d * x + e

def model_3(x, a, b, c, d, e):
    """a*cos(x) + b*sin(x) + c*x^2 + d*x + e"""
    return a * np.cos(x) + b * np.sin(x) + c * x**2 + d * x + e

def model_4(x, a, b, c, d, e, f):
    """a*sin(b*x+c) + d*x^2 + e*x + f"""
    return a * np.sin(b * x + c) + d * x**2 + e * x + f

def model_5(x, a, b, c):
    """a*sin(x) + b*x^2 + c"""
    return a * np.sin(x) + b * x**2 + c

def model_6(x, a, b):
    """a*sin(x) + b*x^2"""
    return a * np.sin(x) + b * x**2

def model_7(x, a, b, c, d):
    """a*sin(x) + b*cos(x) + c*x^2 + d"""
    return a * np.sin(x) + b * np.cos(x) + c * x**2 + d

def model_8(x, a, b, c, d, e):
    """a*sin(x) + b*x^3 + c*x^2 + d*x + e"""
    return a * np.sin(x) + b * x**3 + c * x**2 + d * x + e

models = [
    ("sin(x)+x^2+x+c", model_1, [1, 0.1, 0, 0]),
    ("sin(bx)+x^2+x+c", model_2, [1, 1, 0.1, 0, 0]),
    ("cos+sin+x^2+x+c", model_3, [0, 1, 0.1, 0, 0]),
    ("sin(bx+c)+x^2+x+c", model_4, [1, 1, 0, 0.1, 0, 0]),
    ("sin(x)+x^2+c", model_5, [1, 0.1, 0]),
    ("sin(x)+x^2", model_6, [1, 0.1]),
    ("sin+cos+x^2+c", model_7, [1, 0, 0.1, 0]),
    ("sin(x)+x^3+x^2+x+c", model_8, [1, 0, 0.1, 0, 0]),
]

def compute_bic(y_true, y_pred, k):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    sigma2 = rss / n
    bic = n * np.log(sigma2) + k * np.log(n)
    return bic

results = []
print(f"{'Model':<25} {'k':>3} {'MSE':>12} {'BIC':>12} {'Params'}")
print("-" * 80)

for name, func, p0 in models:
    try:
        popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
        y_pred = func(x, *popt)
        mse = np.mean((y - y_pred)**2)
        k = len(popt)
        bic = compute_bic(y, y_pred, k)
        results.append((name, func, popt, k, mse, bic))
        params_str = ", ".join(f"{p:.6f}" for p in popt)
        print(f"{name:<25} {k:>3} {mse:>12.8f} {bic:>12.4f} [{params_str}]")
    except Exception as e:
        print(f"{name:<25} FAILED: {e}")

# Sort by BIC
results.sort(key=lambda r: r[5])
print(f"\nBest model by BIC: {results[0][0]}")
print(f"  Parameters: {results[0][2]}")
print(f"  MSE: {results[0][4]:.8f}")
print(f"  BIC: {results[0][5]:.4f}")

best_name, best_func, best_popt, best_k, best_mse, best_bic = results[0]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data + best fit
ax = axes[0, 0]
x_plot = np.linspace(-5, 5, 500)
ax.scatter(x, y, s=5, alpha=0.5, label='Training data')
ax.plot(x_plot, best_func(x_plot, *best_popt), 'r-', lw=2, label=f'Best: {best_name}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Best Model: {best_name}')
ax.legend()

# Plot 2: Residuals
ax = axes[0, 1]
y_pred_best = best_func(x, *best_popt)
residuals = y - y_pred_best
ax.scatter(x, residuals, s=5, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('x')
ax.set_ylabel('Residual')
ax.set_title('Residuals (Best Model)')

# Plot 3: BIC comparison
ax = axes[1, 0]
names = [r[0] for r in results]
bics = [r[5] for r in results]
colors = ['green' if i == 0 else 'steelblue' for i in range(len(results))]
ax.barh(range(len(results)), bics, color=colors)
ax.set_yticks(range(len(results)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('BIC')
ax.set_title('Model Comparison (BIC, lower is better)')

# Plot 4: All fits overlay
ax = axes[1, 1]
ax.scatter(x, y, s=5, alpha=0.3, label='Data')
for name, func, popt, k, mse, bic in results[:4]:
    ax.plot(x_plot, func(x_plot, *popt), lw=1.5, label=f'{name} (BIC={bic:.1f})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Top 4 Model Fits')
ax.legend(fontsize=7)

plt.tight_layout()
fig_dir = os.path.dirname(__file__) + "/figures"
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(fig_dir + "/model_comparison.png", dpi=150)
print(f"\nFigure saved to {fig_dir}/model_comparison.png")
