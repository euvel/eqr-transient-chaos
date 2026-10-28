# S2: Lyapunov spectra on EqR saddle trajectories: rough vs filamentary (laptop CPU, true float64, 2026-10-27)
Pre-registration: `PREREG_cross_boundary_eqr.md` (2 dated amendments). Post-hoc rule: `POSTHOC_replicate.md` (written
before the replicate ran). Code: `s2b_stagger.py`, `s2c_benettin_along.py`, `s2_analyze.py`. Raw: `../eqr/data/` (spectra; the stored latent states, ~280 MB, are not included: regenerate with `s2b_stagger.py`).

## Data
Stagger-and-step saddle trajectories, K = 16 Benettin twins (1e-7), window loops [6, end).
| puzzle | kappa | window loops | kicks (sizes) | base-trajectory check |
|---|---|---|---|---|
| 206 | 0.054 | 66 | 0 | 4e-13 |
| 814 | 0.118 | 54 | 4 (1e-7..4e-5) | (off-by-one in kick log, see below) |
| 225 | 0.142 | 36 | 4 (1e-7..1e-3) | (same) |
| 80 | 0.246 | 36 | 7 (3e-7..3e-5) | 0.0 |
| 206 replicate (2nd-slowest pixel) | 0.054 | 48 | 4 | 1.7e-13 |
The kick-log off-by-one affected only the mismatch diagnostic (fixed before the 80 run; 80 and the replicate then show
exact or rounding-level agreement). Exponents are unaffected, because the base is re-synced to the stored state every loop.

## Spectra (lambda_1..lambda_12; SE 0.02-0.17)
| | l1 | l2 | l3 | l4 | l5 | l6 | l7 | l8 | l9 | l10 | l11 | l12 | # pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 206 | 2.23 | 1.68 | 1.26 | 1.01 | 0.66 | 0.46 | 0.48 | 0.35 | 0.19 | 0.13 | 0.05 | 0.01 | 11 |
| 206 rep | 2.33 | 1.68 | 1.29 | 0.92 | 0.84 | 0.55 | 0.29 | 0.37 | 0.25 | 0.07 | 0.02 | -0.05 | 9 |
| 814 | 2.12 | 1.38 | 0.92 | 0.77 | 0.51 | 0.45 | 0.17 | 0.05 | 0.06 | -0.09 | -0.11 | -0.19 | 7 |
| 225 | 1.93 | 1.48 | 1.05 | 0.88 | 0.75 | 0.58 | 0.40 | 0.36 | 0.16 | 0.08 | 0.06 | 0.00 | 9 |
| 80 | 2.34 | 1.52 | 1.19 | 0.79 | 0.74 | 0.45 | 0.33 | 0.25 | 0.21 | 0.08 | 0.00 | -0.08 | 9 |

## Registered verdict
- Stage 1: lambda_2 > 0 on all 4 puzzles (7-11 positive exponents). Rough boundaries are **not** excluded.
- Stage 2: all 4 resolved (lambda_16 < kappa). "Exponent near kappa" holds for all 4, as expected trivially in a dense
  spectrum. The positive spectra do not agree (counts 11/7/9/9), with or without the kappa-matched exponent removed.
  **Verdict: NOT-SKEW.**

## Post-hoc replicate (rule fixed in advance)
The two 206 trajectories agree element-wise (0 of 16 exponents outside tolerance; max |diff| 0.19), but their counts of
positive exponents differ (11 vs 9) because exponents near zero cross the significance threshold. Under the written rule
they disagree, so **the verdict is NOT-SKEW (noise-limited)**: with trajectories of 36-66 loops the method cannot resolve
puzzle-level differences in the count.

## Descriptive (not registered)
- Element-wise, 206, 206-rep, 225 and 80 are mutually consistent (0-1 exponents outside tolerance; mean |diff| over the
  top 10 is 0.08-0.12), although kappa spans 4.5x. 814 is lower (4-7 exponents outside tolerance against every other run).
  No exponent visibly moves with kappa. But a single inserted exponent would shift the lower spectrum by about one
  spacing (~0.05-0.1), which is below the current resolution.
- Sum of positive exponents 6.4-8.5 per loop vs kappa 0.05-0.25. By Kantz-Grassberger, h_KS = Sum(lambda+) - kappa, so the
  saddle leaks only 0.6-3% of its stretching. Escape is an abrupt "snap" (one loop with 27-30 cells changing, then smooth
  convergence). On the saddle all puzzles look alike (~10 cells per loop, latent step ~0.25).

## Status of the open question
Still open for EqR, but narrowed:
1. Direction-resolved tests cannot work here (S0b: float64 noise ~16*alpha decades; CLV contamination).
2. B&L's necessary condition for roughness holds (many positive exponents).
3. The skew-rough mechanism with a puzzle-independent churn is not supported at the current resolution, and not
   excluded either.
Resolving it would need about 3x smaller SEs: roughly 10x longer saddle trajectories (~500 loops) or ~10 replicates per
puzzle. That is about an hour per puzzle on the laptop CPU, or a GPU run.
