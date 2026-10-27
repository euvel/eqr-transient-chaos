# v6: out-of-sample test of the delayed predictability horizon (written 2026-10-27, before any v6 data)

Origin: v5 (PREREG_v5.md, REPORT_v5_2026-10-26.md) failed its registered bar E1c (61% within 2x, bar 70%); the exponent
matched kappa/lambda1 on all 5 puzzles. Post-hoc (same report, step 1): the error was one-sided (horizon too short), and
one constant delay D added to every horizon fixed the amplitude (D = 1.75 loops; leave-one-puzzle-out 1.75-2.0;
18/18 points within 2x). That fit used the v5 test data, so it is tested here on puzzles v5 never used.

## Law under test (frozen)
Per pair i: t_i = ln(0.1 / d0_i) / lambda1 + D, with **D = 1.75 loops** (frozen); d0_i = pair separation after loop 0
(relative to the state); f_pred = mean_i S(t_i). S, kappa, lambda1 from the v3b / v4-part-0 main maps, unchanged.
Everything else identical to v5 (true float64 with the rms_norm patch, same in-kernel checks, threshold 0.1,
eps 1e-1, 1e-3, 1e-5, 1e-8, 1e-12, N = 48 pairs, same plane seed 0). Pair seed changed 1 -> 2 so every pair is new.

## Puzzles (rule, fixed before any v6 data)
All characterised puzzles NOT used as v5 test puzzles with a valid escape fit (R^2 >= 0.97, as in v5) and
kappa/lambda1 >= 0.05 (below that the delay changes f by < 10%: exp(-kappa D) > 0.91, so the test cannot discriminate):
1570 (kappa/lambda1 0.065, cap 56), 80 (0.116, cap 30), 1982 (0.203, cap 16), 149 (0.703, cap 16).
Excluded by the rule: 1216, 206 (kappa/lambda1 0.022, 0.024); 38, 121 (R^2 0.96, 0.94); v5 test puzzles.
149 lies above v5's 0.4 upper limit; it is included because it is where D matters most (exp(-kappa D) = 0.15).
Puzzle data: puzzles_v6.json, built by make_puzzles_v6.py (the recipe reproduces v5's stored entries for 1570, 1982 exactly).

## Bars
- E0 validity: >= 3 of 4 puzzles complete all 5 eps levels.
- E1 (main), points with f_pred (D = 1.75) in [0.05, 0.95], f_obs = (k + 0.5)/(N + 1), same as v5:
  (a) Spearman >= 0.8; (b) median |ln(f_obs/f_pred)| <= ln 2; (c) >= 70% of points within a factor 2.
  **PASS = E0 and E1(a, b, c).**
- Contrast (reported, no bar): the same E1 with D = 0. Expected to be biased high (f_obs < f_pred) as in v5.
- D3 (descriptive): best single D refitted on v6 data alone (grid 0-5, step 0.25, same loss as the v5 post-hoc).
  Prediction: within [1.0, 2.75].
- E2 (descriptive, as v5): per-puzzle log-log slope of f_obs vs eps (levels with 0.03 < f_obs < 0.97) vs kappa/lambda1.

## Prior contact with v6 puzzles (disclosed)
- Before v5: laptop design run of the v5 kernel on 1570 (eps 1e-8) and 1982 (eps 1e-3), 12 pairs each with pair seed 1:
  f_obs 0.250 vs f_pred 0.266; 0.250 vs 0.428. Not used to fit D. v6 uses pair seed 2, so no pair is shared.
- Pre-push laptop smoke of the frozen v6 kernel (code path only, N = 2, pair seed 2 so these 2 pairs ARE in the run):
  149 @1e-1: 2/2 differ (f_pred 0.165, D=0 0.699); 1982 @1e-3: 0/2 differ (f_pred 0.158). No weight. Gates passed on CPU
  (loop-0 ratio 1000.09; stepping == solve_batch). Bar code checked on v5 data: reproduces the official v5 result (D=0)
  and the post-hoc D=1.75 result exactly.

## Budget
20 jobs; ~11.4 s per cap-loop per job on T4 (v5) -> ~1.9 GPU-h, ~1 h wall on T4 x 2. Quota left ~7 h.
- Expected scoreable E1 points (f_pred in [0.05, 0.95], typical c): ~13 (1570: 5, 80: 4, 1982: 3, 149: 1). 149 gives one
  sharp point at eps 1e-1 (D = 1.75 predicts ~0.18, D = 0 predicts ~0.70).
