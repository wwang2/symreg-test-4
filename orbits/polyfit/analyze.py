"""Polynomial fitting analysis with cross-validation and model selection."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import csv

# Load training data
x_list, y_list = [], []
with open('/tmp/train_data.csv') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        x_list.append(float(row[0]))
        y_list.append(float(row[1]))

x = np.array(x_list)
y = np.array(y_list)

# Fit polynomials of degrees 1-15 with 5-fold cross-validation
degrees = range(1, 16)
cv_means = []
cv_stds = []
bic_vals = []
n = len(x)

for deg in degrees:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    scores = cross_val_score(model, x.reshape(-1, 1), y, cv=5, scoring='neg_mean_squared_error')
    cv_means.append(-scores.mean())
    cv_stds.append(scores.std())

    # Fit on full data for BIC
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    rss = np.sum((y - y_pred) ** 2)
    k = deg + 1  # number of parameters
    bic = n * np.log(rss / n) + k * np.log(n)
    bic_vals.append(bic)

best_deg_cv = degrees[np.argmin(cv_means)]
best_deg_bic = degrees[np.argmin(bic_vals)]
print(f"Best degree by CV: {best_deg_cv} (MSE={min(cv_means):.6f})")
print(f"Best degree by BIC: {best_deg_bic} (BIC={min(bic_vals):.2f})")

# Now try polynomial + sin terms
# Build features: polynomial + sin(x) + cos(x)
from sklearn.linear_model import Ridge

def build_features(x, deg, use_trig=False):
    X = np.column_stack([x**i for i in range(1, deg+1)])
    if use_trig:
        X = np.column_stack([X, np.sin(x), np.cos(x)])
    return X

# Try degrees 1-6 with trig terms
print("\n--- Polynomial + Trig ---")
for deg in range(1, 7):
    X = build_features(x, deg, use_trig=True)
    model = Ridge(alpha=1e-6)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    print(f"  deg={deg} + sin/cos: CV MSE = {mse:.6f}")

# Best model: degree 2 + sin(x) (since target is sin(x) + 0.1*x^2)
# Let's fit and check coefficients
X_best = build_features(x, 2, use_trig=True)
model_best = Ridge(alpha=1e-8)
model_best.fit(X_best, y)
print(f"\nBest model coefficients (deg2 + trig):")
print(f"  intercept: {model_best.intercept_:.6f}")
names = ['x', 'x^2', 'sin(x)', 'cos(x)']
for name, coef in zip(names, model_best.coef_):
    print(f"  {name}: {coef:.6f}")

# Generate figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: CV MSE vs degree
ax = axes[0, 0]
ax.errorbar(list(degrees), cv_means, yerr=cv_stds, marker='o', capsize=3)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Cross-Validation MSE')
ax.set_title('CV MSE vs Polynomial Degree')
ax.set_yscale('log')
ax.axvline(best_deg_cv, color='r', linestyle='--', alpha=0.5, label=f'Best={best_deg_cv}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: BIC vs degree
ax = axes[0, 1]
ax.plot(list(degrees), bic_vals, marker='o')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('BIC')
ax.set_title('BIC vs Polynomial Degree')
ax.axvline(best_deg_bic, color='r', linestyle='--', alpha=0.5, label=f'Best={best_deg_bic}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Data + best fit
ax = axes[1, 0]
x_plot = np.linspace(-5, 5, 500)
X_plot = build_features(x_plot, 2, use_trig=True)
y_plot = model_best.predict(X_plot)
ax.scatter(x, y, alpha=0.3, s=10, label='Training data')
ax.plot(x_plot, y_plot, 'r-', linewidth=2, label='sin(x)+0.1x^2 fit')
ax.plot(x_plot, np.sin(x_plot) + 0.1*x_plot**2, 'g--', linewidth=1, label='True (reference)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Best Fit: Degree 2 + Trig')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Residuals
ax = axes[1, 1]
X_train = build_features(x, 2, use_trig=True)
residuals = y - model_best.predict(X_train)
ax.scatter(x, residuals, alpha=0.3, s=10)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('x')
ax.set_ylabel('Residual')
ax.set_title(f'Residuals (std={np.std(residuals):.4f})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/wujiewang/code/symreg-test-4/.worktrees/polyfit/orbits/polyfit/figures/polynomial_analysis.png', dpi=150)
print(f"\nFigure saved.")
