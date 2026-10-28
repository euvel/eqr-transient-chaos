# Findings: rough vs filamentary basin boundaries in EqR (consolidated, 27-28 Oct 2026)
Every claim points to a script and a report in this folder; the
chronological index is `STATUS_cross_boundary.md`. Scripts are in ../toys/, ../eqr/ and ../capstone/. All toy calibrations had predictions written before running;
amendments and bug fixes are dated and disclosed in each script's docstring.

## 0. The question
Bodai & Lucarini (2020): a basin boundary is ROUGH if the cross-boundary Lyapunov exponent lambda_x on the non-attracting
chaotic set is smaller than the maximal one, and FILAMENTARY if lambda_x = lambda_max. Both give the same uncertainty
exponent alpha = kappa/lambda_max, so our earlier result (alpha = kappa/lambda_1 for EqR, 9 puzzles, rho 0.92) cannot
tell them apart. B&L: fractal dimensions cannot decide it; one must measure lambda_x.

## 1. New measurements on EqR (true float64; loopscape 16b52aa; EqR checkpoint sha256 258b1f0a...)
| finding | numbers | source |
|---|---|---|
| saddle Lyapunov spectrum | 7-11 positive exponents; lambda_1 1.9-2.3; spectra of 206 (kappa .054), 814 (.118), 225 (.142), 80 (.246) | REPORT_S2_eqr_spectrum_2026-10-27.md |
| B&L necessary condition for roughness | lambda_2 > 0 on all 4 puzzles (rough not excluded) | same |
| leak of the saddle | sum of positive exponents 6.4-8.5 per loop vs kappa 0.05-0.25 (Kantz-Grassberger: h_KS ~ Sum lambda+ - kappa; escape = 0.6-3% of the stretching (206: 0.6%, 814/225: 1.8%, 80: 3.1%)) | same |
| puzzle dependence | 814 reproducibly lower in lambda_7..lambda_10 (2 replicates agree; 2-3 exponents outside tolerance vs 206/225); 206/225/80 alike within noise | POSTHOC_replicate.md |
| spectrum test (leave-kappa-out) | registered NOT-SKEW; noise-limited (positive-exponent count 11 vs 9 within one puzzle) | PREREG_cross_boundary_eqr.md, REPORT_S2 |
| escape | an abrupt "snap": one loop with 27-30 cells changing and a ~0.83 relative latent jump, then smooth convergence; on the saddle ~10 cells flip per loop, latent step ~0.25 | PREREG_cross_boundary_eqr.md (Amendment 2) |
| churn location | no invariant token group (leakage >= 0.53 after one loop); contested cells partly self-contained (0.53-0.61 vs random 0.73-0.80); settled cells leak more | REPORT_A4_coupling_2026-10-28.md |
| float64 vs float32 settle times | unrelated for slow pixels (206 slowest pixel: 136 in the float32 map, 17 in float64) | PREREG (Amendment 1) |

## 2. Main result: rough boundaries are computationally hidden
### 2a. Degeneracy (argument + toy evidence; not a proof)
For a skew-rough boundary (Hoelder exponent h = lambda_x/lambda_1), a perturbation with a generic churn component sits
~delta^h from the boundary, and churn noise reaches the decision amplified by the same h. The two cancel, so
every generic-perturbation observable scales as in the filamentary case. Classes tested on calibrated toys:
| class | test | outcome | report |
|---|---|---|---|
| bisection along random lines | S0 | 1/lambda_1 in both (0.87 vs 0.92) | REPORT_S0_2026-10-27.md |
| spectrum tracking / leave-kappa-out | S0c, S0d, S2 | works on toys; noise-limited and puzzle-dependent on EqR | REPORT_S0bcd, REPORT_S2 |
| survival-stretch regression | S4 | geometric confounds (FILAM even flips sign) | REPORT_S4_2026-10-28.md |
| topology of 2-D slices (islands vs stripes) | A3 | skew roughness is 1-D at fine scales -> stripes in both | REPORT_A3_islands_2026-10-28.md |
| escape-triggered averages | S5 | relaxation confounds; a FILAMENTARY toy faked a rough rate ~kappa | REPORT_S5_eta_2026-10-28.md |

### 2b. Only exact invariant directions reveal roughness, and they cost precision
- Direction-resolved probing works with the exact cross direction (S0) and with clean long-trajectory CLVs in exact-skew
  toys (S0f: lambda_x 0.406 vs 0.405).
- Rounding law: noise floor delta_floor ~ eps_mach^h; slope fits 0.258/0.393/0.658 vs h 0.239/0.369/0.631 (3/3,
  float16 to 40 digits). REPORT_A_hidden_roughness_2026-10-28.md
- Contamination law: delta_floor ~ c^(h/(1-h)) for a direction error c; 0.587 vs 0.585, 1.603 vs 1.710 (2/3); third
  0.427 vs 0.314 (floors fall faster, the favourable side). Same report.

### 2c. The orbit obstruction and its resolution
- With weak feedback from the decision into the churn (eps = 1e-3), CLVs along kicked float64 stagger orbits lose the
  cross signal (slope 2.46 -> 1.01, verdict wrongly FILAM). REPORT_S0g_gateB_2026-10-28.md
- Genuine kick-free orbits (80-digit bisection, ~160 steps) restore it: 9/9 verdicts ROUGH, lambda_x within 25% in 8/9
  (0.42-0.56 vs 0.405 with feedback). REPORT_S0h_true_orbit_2026-10-28.md

### 2d. Complete cost law
Deciding with a one-decade probe window requires c <= 0.01^((1-h)/h); a genuine saddle orbit of length
L ~ L_warm + ln(1/c)/gap; and ~L * lambda_1 / ln 10 digits. Toy check: 76 digits predicted, 80 used, worked.
EqR puzzle 80 (lambda_1 2.34, h ~0.105 if skew-rough; measured gaps ~0.05-0.1): a genuine orbit of ~450-850 loops in
~460-860-digit arithmetic. Decidable in principle, infeasible in practice.

### 2e. Why the direct routes fail for EqR in practice
- Precision-survival conflict (S3): the float64 window needs fast puzzles, but stagger-and-step only holds slow ones
  (1982 failed with look-ahead 16 and 8). PREREG_S3_feasibility.md
- With double-double arithmetic alone, B (puzzle 80) would still fail on the orbit obstruction (S0g). Stopped at the gate.

## 3. Methods and tools built (reusable)
- Exact forward-mode JVPs through EqR: EqR's inner forward ends with .detach(), which strips tangents; calling
  m.inner.deep_recursion directly keeps them (bit-identical primal; central differences converge to it as eps^2). s3a_jvp.py
- Stagger-and-step for EqR with backtracking and forced kicks (s2b_stagger.py); Benettin along stored trajectories
  (s2c_benettin_along.py); exact-JVP Benettin + Ginelli (s3c_clv_ad.py); direction probe (s3d_probe.py).
- v7 GPU kernel (prepared and smoke-tested, never run; not included): 4 puzzles x 2 replicates x 300 loops spectra; its most likely
  outcome (NOT-SKEW) would not settle the question.
- Calibrated toy suite: S0-S0h, A1-A4, S4, S5 (rough, filamentary, dense-spectrum, feedback, fluctuating-stretch toys).
- Laptop costs: float64 EqR ~0.5 s per sample-loop on CPU; T4 float64 ~8.3 sample-loops/s per GPU (v6).

## 4. What it is worth (assessment)
- Answers B&L's own question ("can we classify them experimentally?") for high-dimensional systems with a quantitative
  "only at this price" plus an explicit price for a trained reasoning model. To our knowledge new: the two precision
  laws, the orbit obstruction, the cost law, and every saddle measurement on a reasoning model.
- Limits: the degeneracy is an argument, not a theorem; the laws are verified on toys of 2-30 dimensions; the EqR price
  assumes skew-rough (h = alpha); one model, Sudoku only.
- Capstone attempted 10-28 (a small trained reasoner where the cost law is affordable): STOPPED at G1. Three tiny looped
  reasoners (incl. a miniature EqR with inner cycles, 94-95% accurate) show contraction or neutral oscillation
  (lambda_1 ~ 0), never transient chaos. Chaos as in EqR seems to need scale. ../capstone/REPORT_G1_2026-10-28.md
