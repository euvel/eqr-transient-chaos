# S0h: kick-free true orbits rescue the cross-direction probe under feedback (toys, mpmath 80 digits, 2026-10-28)
Script `s0h_true_orbit.py` (prediction in the docstring; one probe-sampling bug fixed, disclosed), raw `s0h_true_orbit.json`.
3-D rough toy with feedback x1' += eps tanh(y). Genuine orbits of ~160 steps found by 80-digit bisection (no kicks);
CLVs at step 60 from ~90 backward steps; lifetime sampled at offsets +-w (w = 1e-1..1e-7) along CLV1-3.

| eps | lambda_x_hat (3 orbits) | verdicts | S0g (kicked, float64) |
|---|---|---|---|
| 0 | 0.407, 0.405, 0.406 | ROUGH x3 | 0.406 ROUGH |
| 1e-3 | 0.494, 0.440, 0.454 | ROUGH x3 | 0.99 **FILAM** |
| 1e-2 | 0.559, 0.494, 0.422 | ROUGH x3 | 0.73 ROUGH (+80%) |
Truth lambda_x = 0.405. Prediction (within 25% for every eps): 8/9 (one +38%); verdict correct 9/9. With feedback the
estimate is biased upward by 4-38%.

**Conclusion:** S0g's failure came from the pseudo-orbit (kicks and float64 rounding), not from the probe principle.
With a genuine orbit long enough for the CLVs to converge, the cross direction, and lambda_x, can be recovered even
with feedback.

## The complete cost law
Deciding rough vs filamentary with a window of one decade below width 0.1 needs:
1. direction error c <= 0.01^((1-h)/h) (contamination law, REPORT_A);
2. a genuine orbit on the saddle of length L ~ L_warm + ln(1/c)/gap (backward CLV convergence; gap = spacing between
   the cross exponent and the next faster one);
3. arithmetic with ~ L * lambda_1 / ln 10 digits, so that orbit can be held without kicks (rounding law is then automatic).
Toy check: L = 160, lambda_1 = 1.1 -> ~76 digits (used 80; it worked).
EqR puzzle 80 (lambda_1 2.34, h ~0.105 if skew-rough): c <= 1e-17;
| spectral gap | orbit length | digits |
|---|---|---|
| 0.05 | ~840 loops | ~850 |
| 0.1 | ~450 loops | ~460 |
| 0.2 | ~255 loops | ~260 |
EqR's measured spectrum has gaps of ~0.05-0.1 in the relevant range, so deciding needs a genuine 450-850-loop saddle
orbit of a 99k-dimensional transformer in ~460-850-digit arithmetic. That is far beyond reach: for EqR the question is
decidable in principle and infeasible in practice, with the price stated exactly.
