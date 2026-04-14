---
issue: 2
parents: []
eval_version: eval-v1
metric: 1.43e-32
---

# Research Notes

## Hypothesis
Fit polynomials of varying degrees to the training data. Use cross-validation and BIC to select the best degree. Establish a polynomial baseline, then augment with trig terms.

## Approach
1. Dumped 200 noisy training points (x in [-5,5], noise sigma=0.05)
2. Fit polynomials of degrees 1-15, evaluated with 5-fold CV and BIC
3. Best pure polynomial: degree 7 (CV MSE=0.002475)
4. Augmented with sin/cos basis functions: degree 2 + trig achieved CV MSE=0.002497
5. Inspected coefficients of degree-2 + trig model:
   - sin(x): 1.007, x^2: 0.100, all others near zero
6. Concluded: f(x) = sin(x) + 0.1 * x^2

## Result
- **metric**: 1.43e-32 (mean MSE across seeds 1,2,3)
- Exact symbolic form recovered: `sin(x) + 0.1 * x^2`

## Key Findings
- Pure polynomial (degree 7) approximates well but cannot match exact form
- Adding sin/cos basis to low-degree polynomial immediately reveals the true structure
- Coefficient inspection makes the symbolic form obvious
