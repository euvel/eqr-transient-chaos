# v3: does boundary roughness equal escape rate / Lyapunov exponent? (written 2026-10-26, before any v3 data)

Origin: POST-HOC observation in v2 (REPORT_v2_2026-10-26.md). For P1 (two planes), the lab's roughness exponent
H (loopscape.fractal.variogram_H) was 0.35 and 0.27 while kappa/lambda1 was 0.26 and 0.30; for slow puzzles
(HARD, P3) kappa/lambda1 ~ 0.03 and 0.006 and the lab pipeline returned "decorrelated" (H = NaN). Chaotic-saddle
theory relates boundary roughness to escape rate over Lyapunov exponent. v2 points are NOT used in this test.

## Fixed inputs
- Candidates: kaggle_v3/candidates_v3.json = Sudoku-Extreme test.csv in file order: first 24 with rating >= 90
  plus first 24 with 50 <= rating < 90, excluding the 5 puzzles used in v1/v2 (48 candidates).
  (Changed before any real v3 data: the first draft used the first 32 with rating >= 50; a CPU toy dry run of its
  first 4 candidates showed median settle 0-2 loops, i.e. no transient, and v2's transients came from ratings 112-125.)
- Model/plane/observable exactly as v2: EqR (loopscape), float32, demo 2-plane seed 0, z_L = 0, noise 0,
  T_dec = loops until the decoded grid never changes again.

## Procedure (all automatic inside one kernel)
1. Screen: every candidate (48) on a 16x16 map, cap 100 -> kappa_screen (same escape fit as v2).
2. Select: candidates with a valid kappa_screen in [0.05, 1.5]. Sort by kappa_screen; if more than 8 qualify,
   take 8 evenly spaced by rank (indices round(linspace(0, n-1, 8))) to span the range. Screen data are used
   only for selection.
3. Main map per selected puzzle: 40x40, cap 150 -> kappa (escape fit on this map), lab field_stats on T_dec
   (H, alpha, D, saturation, R^2).
4. Lyapunov per selected puzzle: the 8 slowest points of the main map, K = 4, cap 100, 2-loop burn-in, window =
   each trajectory's own settle; points without a window excluded. lambda1 = median over included points.
5. Test set = selected puzzles with finite H, kappa and lambda1.

## Bars
- B0 (validity) at least 5 puzzles in the test set; otherwise the test is UNDETERMINED (no pass, no fail).
- B1 Spearman rho(H, kappa/lambda1) >= 0.7.
- B2 median |H - kappa/lambda1| <= 0.10.
PASS = B0 and B1 and B2. Reported alongside, no bars: the same for alpha (lab uncertainty_alpha); Spearman of H
with kappa alone (lambda1 may be near-constant, so B1 alone largely tests kappa; B2 is the part that needs lambda1);
puzzles whose H is NaN, with their kappa/lambda1 (prediction: small).
Time guard: no new phase after 10 h.

## Amendment 1 (2026-10-26): v3 cancelled after 20 min of screening, before any main map or H value existed
Seen: screening only (7 candidates: 2 with median settle 0, 4 with 1-3, 1 with 99; no kappa fittable).
Problem: the 16x16 screen cannot fit kappa for fast puzzles (the fit needs >= 5 loops of decay), so selection
excluded exactly the fast-but-transient puzzles whose H is measurable, and favoured slow puzzles whose predicted
H ~ kappa/2.1 is near 0 (reported NaN by the lab pipeline). Redo = kernel sidep01-eqr-v3b with:
- screen: 16x16, cap 60 (was 100);
- selection on transient existence, not on a fitted kappa: median T_dec in [2, 30] and at most 10% of points
  still changing at the cap. Sort by median settle (ties by row); if more than 8 qualify, take 8 evenly spaced
  by rank (indices round(linspace(0, n-1, 8)));
- escape-rate fit minimum span relaxed from 5 to 3 loops (screen and main map; nothing else in the fit changes).
Main maps, Lyapunov, test set and bars B0-B2 unchanged.
