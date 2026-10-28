# Post-hoc check (not pre-registered): within-puzzle replicate, written before the replicate is run (2026-10-27)
Why: the Stage-2 tolerances use the within-trajectory block-bootstrap SE. If two trajectories of the SAME puzzle disagree
as much as different puzzles do, then the registered NOT-SKEW reflects trajectory-to-trajectory variability (a
non-hyperbolic saddle or finite-time effects), not puzzle-dependent spectra.
Run: puzzle 206, second-slowest pixel (STAGGER_RANK=1), same stagger + Benettin settings (K = 16).
Rule: apply the same Stage-2 agreement statistic (full positive spectra, tol = max(0.08, 3 sqrt(SE1^2 + SE2^2))) to the
two 206 trajectories.
- If they agree, the between-puzzle differences are real, and NOT-SKEW means the spectrum genuinely depends on the puzzle.
- If they disagree, the method cannot resolve puzzle-level spectral differences with ~40-70-loop trajectories, and the
  verdict must be reported as NOT-SKEW (noise-limited).

## Second replicate: puzzle 814 (written before it runs, 2026-10-27) - decides whether v7 is worth the GPU
Run: 814, second-slowest pixel (STAGGER_RANK=1), same laptop settings (stagger with Amendment 2, Benettin K = 16).
Compare the element-wise top 12 with (a) the first 814 run and (b) the pooled 206/206-rep/225/80 runs, using the same
tolerance max(0.08, 3 sqrt(SE1^2 + SE2^2)) and counting exponents outside it.
- 814-rep closer to 814 (<= 1 outside) AND >= 4 outside against at least 3 of the 4 other runs -> the 814 difference is
  real -> v7 would very likely return NOT-SKEW -> recommend NOT spending GPU on the open question.
- 814-rep within tolerance of the other runs (<= 1 outside against at least 3 of 4) -> the first 814 run was atypical ->
  recommend pushing v7.
- Anything else -> mixed; report both and let the user decide.

### Outcome (814 replicate, 48 window loops, 5 kicks, base mismatch 1.7e-13)
Exponents outside tolerance (top 12): vs 814 = 0; vs 206 = 3, 206-rep = 2, 225 = 2, 80 = 0.
Rule: neither branch holds (not >= 4 vs >= 3 others; not <= 1 vs >= 3 others) -> **MIXED**, user decides.
Reading: 814 is reproducible (the two runs agree exactly within tolerance) and sits lower in lambda_7..lambda_10 than
206/225 (lambda_8 0.05-0.15 vs 0.25-0.37). A modest but real puzzle effect is likely; longer v7 trajectories would make it
significant, so v7's most likely verdict is NOT-SKEW (puzzle-dependent spectra), which would not settle rough vs filamentary.
