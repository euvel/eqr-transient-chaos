# v5: true-float64 eps-pair test of the predictability-horizon law (written 2026-10-26, before any v5 Kaggle run)

Hypothesis: sidep01/HYPOTHESIS_predictability_horizon_2026-10-26.md, corrected by the local deep dive
(sidep01/local_horizon/REPORT_local_horizon_2026-10-26.md): two starting thoughts at relative distance eps share their
settle time unless thinking outlasts the horizon t = ln(0.1 / d0) / lambda1, where d0 is the pair's separation after
loop 0 (relative to the state). Hence the fraction of pairs with different settle times is f(eps) ~ mean_i S(t_i),
with S = the puzzle's survival curve (fraction still thinking) and lambda1 = its Lyapunov rate, both measured earlier.
The law passed one pre-registered local test (EASY, 3 eps levels, 24 pairs). v5 tests it on new puzzles.

## Fixed inputs
- Model: EqR (loopscape), TRUE float64: forward_dtype float64 AND upstream rms_norm patched to keep the input precision
  (it casts to float32 internally). In-kernel checks (abort if failed): loop-0 separation ratio d(1e-3)/d(1e-6) = 1000
  within 1%; own stepping loop == loopscape solve_batch in this configuration.
- Puzzles (rule, fixed before any v5 data): from the v3b/v4-part-0 puzzles with a valid kappa (escape fit R^2 >= 0.97)
  and kappa/lambda1 in [0.05, 0.4] -> 8 candidates (814, 1570, 225, 80, 102, 265, 1982, 2045); 5 taken evenly by rank
  of kappa/lambda1 (indices 0, 2, 4, 5, 7): 814, 225, 102, 265, 2045 (kappa/lambda1 0.060, 0.068, 0.159, 0.194, 0.355).
- kappa, lambda1, S(t): from the v3b / v4-p0 main maps (float32, 40x40, cap 150), embedded in the kernel.
- eps levels: 1e-1, 1e-3, 1e-5, 1e-8, 1e-12. N = 48 pairs per (puzzle, eps). Base points uniform on the demo plane
  (seed 0), partner = base + eps*||base||*w (w random unit direction in the full latent), same batch.
- Cap per puzzle: max(16, ceil(1.5 x 99th percentile of the map's settle times)): 64, 50, 21, 20, 16.
- Decorrelation threshold 0.1 (set on EASY; not tuned on any v5 puzzle).

## Prediction (computed in-kernel, before comparing)
Per pair i: t_i = ln(0.1/d0_i)/lambda1 (d0_i measured at loop 0, which contains no settle-time information);
f_pred = mean_i S(t_i). Variants reported, not used for bars: puzzle-median c; fixed c = 0.039 (EASY).

## Bars
- E0 validity: >= 4 of 5 puzzles complete all 5 eps levels.
- E1 (main), over all (puzzle, eps) points with f_pred in [0.05, 0.95], using f_obs = (k + 0.5)/(N + 1):
  (a) Spearman(f_pred, f_obs) >= 0.8; (b) median |ln(f_obs/f_pred)| <= ln 2; (c) >= 70% of points within a factor 2.
  PASS = E0 and E1(a,b,c).
- E2 (descriptive): per-puzzle log-log slope of f_obs over levels with 0.03 < f_obs < 0.97, compared with kappa/lambda1;
  Spearman of slopes vs kappa/lambda1 across puzzles.
Points with f_pred < 0.05 are reported (Poisson regime) but not in E1.

## Design data seen before freezing (not part of any result)
EASY local runs (T1, T1b). Laptop c-check on 814, 225, 102, 265, 2045 at loop 0 ONLY (perturbation-shrink factor c;
no settle times). A local design run of this exact kernel on NON-test puzzles 1982 (eps 1e-3) and 1570 (eps 1e-8),
12 pairs each, to verify the code path end to end.

## Budget
~25 jobs, ~82k float64 sample-loops; at ~8/s per T4 x 2 GPUs ~ 1.5 h (time guard 10 h). Remaining Kaggle quota ~10 h.
- Final code-path smoke of the frozen kernel: test puzzle 2045 at eps 1e-1 with N = 2 pairs (1 of 2 differed). This
  number has no weight and is disclosed only because it touched a test puzzle.
- Local design results (non-test puzzles, 12 pairs): 1570 @ 1e-8 f_obs 0.250 vs f_pred 0.266; 1982 @ 1e-3 f_obs 0.250
  vs f_pred 0.428. Not part of any result.
- Kernel pins loopscape @ 16b52aa1e9382f8cdd7af45a1ba7eb1efa755b3d and aborts unless the EqR checkpoint sha256 is
  258b1f0a90003a2db0de0c8df7831683ffce00de295ed4bba15dc8e599920d26.
