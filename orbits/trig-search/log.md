---
issue: 3
parents: []
eval_version: eval-v1
metric: 1.43e-32
---

# Research Notes

## Hypothesis
Systematic combinations of trigonometric functions (sin, cos) with polynomial terms using scipy curve_fit, with BIC for model selection.

## Method
- Dumped 200 noisy training points from the evaluator (seed 42)
- Fit 8 candidate models combining sin, cos, and polynomial terms via scipy curve_fit
- Compared models using BIC (Bayesian Information Criterion)
- Best model by BIC: `sin(x) + 0.1 * x^2` (2 parameters, BIC = -1182.52)

## Candidate Models Tested
| Model | k | Train MSE | BIC |
|-------|---|-----------|-----|
| sin(x) + b*x^2 | 2 | 0.00257 | -1182.52 |
| a*sin(x) + b*x^2 + c*x + d | 4 | 0.00247 | -1179.86 |
| a*sin(x) + b*x^2 + c | 3 | 0.00257 | -1177.23 |
| a*cos(x) + b*sin(x) + c*x^2 + d*x + e | 5 | 0.00245 | -1176.04 |
| a*sin(bx) + c*x^2 + d*x + e | 5 | 0.00247 | -1174.56 |
| a*sin(x) + b*x^3 + c*x^2 + d*x + e | 5 | 0.00247 | -1174.56 |
| a*sin(x) + b*cos(x) + c*x^2 + d | 4 | 0.00255 | -1173.17 |
| a*sin(bx+c) + d*x^2 + e*x + f | 6 | 0.00245 | -1170.74 |

## Solution
```python
f(x) = sin(x) + 0.1 * x^2
```

## Results
| Seed | MSE |
|------|-----|
| 1 | 1.46e-32 |
| 2 | 1.43e-32 |
| 3 | 1.41e-32 |

Mean MSE: 1.43e-32

## Conclusion
The simplest model (fewest parameters) won by BIC, recovering the exact generating function `sin(x) + 0.1*x^2`. The near-zero test MSE confirms an exact match.
