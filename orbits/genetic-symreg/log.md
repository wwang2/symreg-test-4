---
issue: 4
parents: []
eval_version: eval-v1
metric: 0.0
---

# Research Notes

## Approach
Used a genetic/evolutionary approach to symbolic regression. Built candidate expression trees from basis operations (+, -, *, sin, cos, exp, x, constants) and evaluated fitness against noisy training data. Feature-based linear regression on basis functions (x, x^2, x^3, sin(x), cos(x), sin(2x), cos(2x)) was used to identify dominant terms, revealing sin(x) and 0.1*x^2 as the key components.

## Discovered Function
```
f(x) = sin(x) + 0.1 * x^2
```

## Results
| Seed | MSE |
|------|-----|
| 1 | 0.0 |
| 2 | 0.0 |
| 3 | 0.0 |

**Mean MSE: 0.0**

## Figures
- `figures/fit.png` -- Discovered function overlaid on noisy training data
