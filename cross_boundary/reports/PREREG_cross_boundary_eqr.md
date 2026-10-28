# Pre-registration: rough vs filamentary in EqR via the Lyapunov spectrum (laptop, 2026-10-27)

Written while the first S2 run (puzzle 206, K = 16) was computing and **before any EqR spectrum beyond lambda_1 was
seen**. Calibration: S0 (`REPORT_S0_2026-10-27.md`), S0b, S0c, S0d (`REPORT_S0bcd_2026-10-27.md`).

## Why the spectrum, not directions
S0b showed that probing the boundary along a direction cannot work for EqR. Float64 noise leaves about 16*alpha usable
decades. The CLVs of weaker exponents are also contaminated by faster ones (the gap-limited backward convergence of a
transient trajectory), and the Hoelder roughness amplifies that contamination. The spectrum itself is robust (forward QR).

## Data
- True float64 EqR (both precision patches), slice seed 0, starting pixel = slowest pixel of the saved 40x40 map.
- Benettin twins + QR, K = 16, relative offset 1e-7, renormalised every loop (`s2_spectrum.py`).
- Puzzles (all with float32-map slow trajectories >= 30 loops): 206 (kappa 0.054), 814 (0.118), 225 (0.142), 80 (0.246).
  Optional: 1570 (0.122) and 1216 (0.049). One trajectory per puzzle, each run <= 15 min.
- If a pixel's float64 settle time is < 30 loops, the next-slowest pixel is used (chaos makes float64 settle times
  differ from the float32 map). Every pixel tried is reported.

## Estimates
- Window: loops [6, T64 - 6) of the base trajectory. At least 24 loops are required.
- lambda_k = mean of the local exponents in the window; SE from a block bootstrap (blocks of 8 loops, 2000 resamples).
- kappa_i from the saved maps (float32). Its CI95 is reported alongside.

## Stage 1: necessary condition (decisive on its own if it fails)
Rough requires a second positive exponent on the saddle. A puzzle has a positive lambda_k if
lambda_k > 0.03 and lambda_k > 2 SE.
- If lambda_2 is not positive on every tested puzzle: **verdict FILAMENTARY** (B&L's rough condition fails).

## Stage 2: leave-kappa-out (only if Stage 1 allows rough)
A puzzle is *resolved* if lambda_K < kappa_i - 0.03, meaning the K computed exponents reach below kappa_i.
Unresolved puzzles are reported as "K too small"; K is then raised (resumable runs) before any verdict.
On >= 3 resolved puzzles:
- m_i = the exponent nearest kappa_i; ok_i iff |m_i - kappa_i| <= max(0.1 kappa_i + 0.03, 2 SE).
- Pairwise tolerance tol_ij = max(0.08, 3 sqrt(SE_i^2 + SE_j^2)), compared element-wise on sorted positive spectra.
- **ROUGH-SKEW**: all ok_i, and the spectra with m_i removed agree within tol (same count of positive exponents).
- **FILAM-CONSISTENT**: the full positive spectra agree within tol (kappa is not an exponent; the spectrum does not move).
- **NOT-SKEW**: otherwise.

## What each verdict does and does not mean
- FILAMENTARY (Stage 1) answers the open question for EqR on these puzzles.
- ROUGH-SKEW: strong evidence of rough boundaries with a skew (churn drives decision) structure, alpha = lambda_x/lambda_1.
- FILAM-CONSISTENT excludes the skew-rough mechanism, but **not** rough boundaries without skew structure. It must be
  reported as such.
- NOT-SKEW: the open question stays open. We report the spectra as they are.

## Known limitations (stated in advance)
- One trajectory per puzzle, finite length (~30-100 loops). Exponents are finite-time values.
- A dense spectrum may make a nearest-exponent match near kappa coincidental. That is why removal-residual agreement
  across puzzles is required, not just a match.
- kappa comes from float32 maps, while the spectrum is float64. Survival statistics were shown to be robust in v2.

## Amendment 1 (2026-10-27, after the first S2 run; decision rules unchanged)
The first run (puzzle 206, slowest float32 pixel) settled at loop **17** in float64 (float32 map: 136); chaos makes
float64 settle times unrelated to the float32 map. With 5 window loops it is unusable (24 required). Disclosure: its
5-loop spectrum was looked at (exploratory only): about 9-10 positive local exponents (2.3, 1.8, 1.25, 0.95, ...),
crossing zero near k = 10-13. Float64 trajectories of >= 36 loops are rare (~1 in 500 starts for kappa 0.25), and
bisection cannot reach them below the float64 floor.
Change of trajectory source, applied to all puzzles alike: **stagger-and-step** (Sweet, Nusse & Yorke 2001) from the
slowest pixel (`s2b_stagger.py`: look-ahead 12 loops, advance 6, kicks 1e-10..1e-4 only when the trajectory is about
to settle), followed by Benettin along the stored saddle trajectory (`s2c_benettin_along.py`). The window becomes
loops [6, end); no settle tail. Kick count and sizes are reported per puzzle.

## Amendment 2 (2026-10-27, after the first stagger runs for 814/225/80; no spectrum of these puzzles computed yet)
Stagger-and-step kept 206 chaotic for 73 loops without any kick, but for 814, 225 and 80 it lost the trajectory: escape is
an abrupt "snap" (one loop with 27-30 cells changing and a ~0.83 relative latent jump, then smooth convergence), and when
the look-ahead flags it the trajectory is already committed, so kicks <= 1e-4 cannot rescue it (3/3, 3/3 and 2/5 kicks
failed). Changes (all puzzles alike; 206 is re-run only if its trajectory would differ, which it does not, since it needed
no kick): look-ahead 16 loops, kicks log-uniform in [1e-7, 1e-2], 4 attempts, and **backtracking**: if every candidate
fails, drop the last 6 loops and kick from the earlier state. Kick sizes and backtracks are reported. Window and decision
rules unchanged. Observation recorded for later: on the saddle all four puzzles look alike (~10 cells flipping per loop,
relative latent step ~0.25).
