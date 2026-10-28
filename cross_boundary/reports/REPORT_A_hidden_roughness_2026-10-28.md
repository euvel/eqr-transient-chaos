# A: rough basin boundaries are computationally hidden (theory + toys, laptop, 2026-10-28)
Scripts: `a1_precision_law.py`, `a1b_contamination_law.py` (predictions in the docstrings, written before running;
one detector bug fixed in A1, disclosed). Raw: matching `.json`. Earlier evidence: S0, S0b, S0f, S4.

## 1. Degeneracy: generic perturbations cannot tell rough from filamentary (argument + evidence)
Skew-rough setting: churn with top exponent lambda_1 drives a decision coordinate with exponent lambda_x < lambda_1; the
boundary is a graph over the churn with Hoelder exponent h = lambda_x/lambda_1. Take a perturbation of size delta with
a generic (nonzero) churn component.
- Geometry: its distance to the boundary is ~ delta^h (Hoelder), so the outcome changes after
  ln(1/delta^h)/lambda_x = ln(1/delta)/lambda_1 loops, exactly as for a filamentary boundary crossed transversally.
- Noise: churn noise of size eta reaches the decision coordinate amplified to eta^h, which cancels in the same way.
Every observable built from outcomes under generic perturbations therefore scales as in the filamentary case with the
same kappa and lambda_1: bisection lines, epsilon-pair fractions, rounding or noise sensitivity. This extends Bodai &
Lucarini's statement (fractal dimensions cannot tell) from dimensions to any generic-perturbation observable.
Numerical evidence: random-line bisection gives 1/lambda_1 in both toy classes (S0: 0.87 vs 0.92); the earlier
epsilon-pair law alpha = kappa/lambda_1 holds in both by theory. Survival statistics are not generic-perturbation
observables, but they fail for a different reason, geometric confounding (S4).
Status: an argument with numerical support, not a proof.

## 2. Only exact invariant directions reveal roughness, and they cost precision
Along the exact cross direction the lifetime grows at 1/lambda_x (S0, S0f). Two precision costs set where that
signal drowns:

**Rounding law: delta_floor ~ eps_mach^h** (A1, toy in a rotated basis, float16 to 40 digits, 3 values of h)
| h | fitted slope of log floor vs log eps | prediction |
|---|---|---|
| 0.239 | 0.258 (+8%) | 0.239 |
| 0.369 | 0.393 (+6%) | 0.369 |
| 0.631 | 0.658 (+4%) | 0.631 |
Confirmed 3/3 across 37 decades of machine precision.

**Contamination law: delta_floor ~ c^(h/(1-h))** for a probe direction contaminated by a fraction c of the churn (A1b, 40 digits)
| h | fitted slope | prediction h/(1-h) |
|---|---|---|
| 0.369 | 0.587 | 0.585 |
| 0.631 | 1.603 (-6%) | 1.710 |
| 0.239 | 0.427 (+36%; floors fall faster than predicted, the favourable side) | 0.314 |
Confirmed 2/3; the third misses the 25% band in the direction of better-than-predicted resolution.

Consequence: usable decades ~ min(h * log10(1/eps), h/(1-h) * log10(1/c)) - 1. Both shrink to zero as h -> 0, so
**strongly rough boundaries (small h) are hidden at any fixed precision.** The number of digits needed grows like 1/h.

## 3. What this means for EqR (if skew-rough, h = alpha = kappa/lambda_1)
| puzzle | kappa | h | float64 (eps 2e-16, CLVs c ~ 1e-15) | double-double everywhere (eps 1e-32, c ~ 1e-30) | stagger feasible |
|---|---|---|---|---|---|
| 206 | 0.054 | 0.024 | 0.0 decades | 0.0 | yes |
| 814 | 0.118 | 0.056 | 0.0 | 0.8 | yes |
| 225 | 0.142 | 0.074 | 0.2 | 1.4 | yes |
| 80 | 0.246 | 0.105 | 0.6 | **2.4** | yes |
| 1982 | 0.477 | 0.203 | 2.2 | 5.5 | no |
| 2045 | 0.858 | 0.355 | 4.6 | 10.3 | no |
| 121 | 0.922 | 0.448 | 6.0 | 13.3 | no |
- **In float64 no EqR puzzle is decidable.** Where stagger-and-step can keep a trajectory on the saddle (kappa <= 0.25)
  a rough signal would show over <= 0.6 decades; where the window opens, the saddle cannot be held (S3).
- **With double-double for both the trajectory deviations and the CLVs, puzzle 80 opens to ~2.4 decades.** A rough
  boundary would then give ~1/kappa = 4 loops per e-fold (~22 loops over 2.4 decades) against ~1/lambda_1 = 0.43
  (~2.4 loops). That would be decisive (option B).
- An absent rough signal is informative only inside the usable window; outside it, "no signal" means nothing.

## 4. Link to earlier observations
The v2 finding that batch composition changed 93% of the settle-time pixels on the hardest puzzle is the visible side
of the same fact: when h is small, the fine structure of the outcome lives below the reach of any fixed precision.
