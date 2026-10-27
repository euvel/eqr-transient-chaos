# v4 confirmation of "H = kappa / lambda1": pre-registration
Written 2026-10-26 while v3b was in its main phase (main maps of 8 puzzles running). At writing time NO H value of
v3b existed: the only v3b data seen were screening numbers (kappa_screen, median settle) in the kernel log, which
contain no H. This file is committed to run whatever v3b shows (pass, fail or undetermined).

## Why
v3b tests the relation on at most 8 puzzles. With 5-8 usable puzzles a Spearman of 0.7 has p ~ 0.05-0.19, so a v3b
pass would be suggestive only. v4 is an independent confirmation on fresh puzzles under the identical protocol.

## Puzzle pool (fixed by rule)
All candidates in v3b's screen (kaggle_v3/candidates_v3.json, 48 puzzles) that MET v3b's selection rule
(median screen settle in [2, 30] and <= 10% still changing at the screen cap of 60) but were NOT among v3b's
8 selected puzzles. Per the v3b log: 29 qualified, 8 selected -> 21 puzzles. The list is read from v3b's
report.json "screen" block by this rule; no other filtering.

## Protocol (identical to v3b main phase)
EqR (loopscape), float32, demo 2-plane seed 0, z_L = 0, noise 0; 40x40 map over [-1,1]^2, cap 150, batch 256;
T_dec = suffix settle of the decoded grid; kappa = v3b escape fit (min span 3); H, alpha = loopscape.fractal
field_stats on T_dec (n_max = cap - 1); lambda1 = median over the 8 slowest points (K = 4, cap 100, 2-loop burn-in,
own window, points without a window excluded). Test set = puzzles with finite H, kappa and lambda1.
Execution: two kernels (pool split by alternating rows in file order) to stay inside Kaggle's session limit,
both T4 x2, same code; results pooled.

## Bars (on v4 alone, independent of v3b)
- C0 validity: v4 test set >= 10 puzzles; otherwise UNDETERMINED.
- C1 Spearman rho(H, kappa/lambda1) >= 0.6 AND exact permutation p < 0.01 (10^5 permutations, one-sided).
- C2 median |H - kappa/lambda1| <= 0.10.
CONFIRMED = C0 and C1 and C2.
Reported without bars: the same on v3b + v4 pooled; alpha vs kappa/lambda1; least-squares slope of H on
kappa/lambda1 through the origin (theory: 1); Spearman of H with kappa alone; every puzzle with H = NaN and its
kappa/lambda1.

## Not done in v4 (stated limits)
One plane (seed 0), one model (EqR), one task (Sudoku). A pass would justify, not replace, checks on other planes
and on FPRM.

## Note (2026-10-26, before launch): pool fixed early from the v3b log
The user chose to launch v4 while v3b was still running. The pool rule needs v3b's censored fractions (only in
report.json, available at the end), so the pool was reconstructed from the v3b log: 30 rows had median settle in
[2, 30]; v3b reported 29 qualifying; excluding row 169 (median 21) is the ONLY single exclusion that reproduces
v3b's 8 selected rows under its own rule. Pool (21): kaggle_v3/pool_v4.json. To be re-checked against v3b's
report.json; any mismatch will be reported. Still no v3b H value existed at launch.

## Deviation (2026-10-26): part 1 HELD (user decision), part 0 continues
Part 1 (10 puzzles) could not start at launch (Kaggle limit: 2 concurrent GPU sessions). Before it was pushed, the
first two v3b main results arrived: row 80 kappa 0.246 / H NaN; row 36 kappa None / H 0.43. Both unusable for the
test, for opposite reasons (H not measurable on slower puzzles; kappa not fittable on the fastest). This concerns
MEASURABILITY (C0), not the relation itself: no (H, kappa/lambda1) pair has yet been seen for any puzzle.
Decision: hold part 1 to save ~4-5 GPU-hours until v3b and v4 part 0 show how many puzzles are usable at all.
Consequence for the bars: if part 1 is never run, C0-C2 are evaluated on part 0 alone (11 puzzles) and the
deviation is reported with the result. Part 1 stays ready (kaggle_v3/eqr_v4_p1/), unchanged.

## Outcome (2026-10-26)
Part 0: 2 usable of 11. C0 (>= 10 usable over the pool) cannot be met even with part 1 -> v4 UNDETERMINED.
Part 1 not run. Full numbers: REPORT_v3b_v4p0_2026-10-26.md.
