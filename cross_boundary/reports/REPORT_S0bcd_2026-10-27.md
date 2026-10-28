# S0b-S0d: blind calibration in hidden-basis toys (laptop, numpy, 2026-10-27)
Scripts (expectations and dated amendments in each docstring): `s0b_blind.py`, `s0c_tracking.py`, `s0d_dense.py`.
Raw: the matching `.json` files. Debug trace: `dbg_s0b.py`.

## S0b: direction-resolved probing in 30-D with a hidden basis. FAILED, and it explains why the method cannot work
- B1 (spectrum) passed: ROUGH (1.10, 0.69, 0.39, -0.69, -0.69), FILAM (1.10, 0.69, -0.66, ...).
- B3/B5 failed: the pipeline called ROUGH filamentary in both versions (lambda_x_hat 0.93, then 0.89; truth 0.405).
- **Obstacle 1, float64 noise.** Rounding enters all directions and grows at lambda_1. A cross-direction probe at width
  delta stays readable only while ln(1/delta) < ln(1e16) * lambda_x/lambda_1, i.e. ~16*alpha decades.
  For EqR: ~0.3 decades for slow puzzles (alpha 0.02), ~7 for fast ones (alpha 0.45).
- **Obstacle 2, CLV contamination.** The computed CLV3 kept 3.9% / 0.2% components along the faster churn directions,
  because the backward iteration on a transient trajectory converges only like exp(-gap * remaining length). On a rough
  boundary, any churn component in the probe gives back the 1/lambda_1 slope (Hoelder). Probing along the TRUE cross
  direction gave ~2.3 loops per e-fold (truth 2.47), so the probe logic is right; the direction is not obtainable.
- Consequence: in EqR (dense spectrum, short transients) direction probing is ruled out. This is a methodological result
  in its own right: the experimental test of Bodai & Lucarini's criterion is far harder in high dimensions than in 2-3 D.

## S0c: spectrum tracking (exponents only) on toy families. PASSED
- ROUGH family (kappa 0.18-0.82): an exponent equals kappa in 5/5 members (|d| <= 0.05), Spearman 1.0 -> TRACKS.
- FILAM family (kappa 0.16-1.04): spectrum fixed; 1 coincidental match (kappa 0.67 next to ln2) -> does not track.

## S0d: leave-kappa-out test on dense-spectrum toys (8-D hyperbolic churn, 4 positive exponents incl. pairs). PASSED after Amendment 1
- First run: both families NOT-SKEW. Fast members (longest trajectory 6-22 steps) have biased finite-time spectra; the
  verdict-independent sign is that known stable exponents (-0.69) drifted to -0.36..-0.6 in both families.
- Amendment 1: use only members with T_max >= 30. ROUGH -> ROUGH-SKEW (residual after removal 0.022);
  FILAM -> FILAM (residual 0.029). Three members each.

## Carried into the EqR pre-registration
Stage 1: the necessary condition (a second positive exponent). Stage 2: leave-kappa-out on slow puzzles only
(trajectories >= 30 loops), with tolerances widened to the measured noise. A FILAM-consistent result excludes only the
skew-rough mechanism, not every possible rough boundary.
