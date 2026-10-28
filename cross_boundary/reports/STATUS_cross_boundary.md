# Cross-boundary exponent (rough vs filamentary) for EqR: status log
Consolidated summary: FINDINGS_cross_boundary_2026-10-28.md
Question (Bodai & Lucarini 2020): is EqR's basin boundary rough (cross-boundary exponent lambda_x < lambda_max) or filamentary?
Both give alpha = kappa/lambda_max, so our published exponent law cannot tell them apart.

| Date | Step | Result | File |
|---|---|---|---|
| 10-27 | S0 toys (known directions) | random-line bisection gives 1/lambda_max in both cases; only the special cross direction shows lambda_x | REPORT_S0_2026-10-27.md |
| 10-27 | S0b hidden-basis toys | direct probe fails: float64 noise (~16*alpha decades) + CLV contamination | REPORT_S0bcd_2026-10-27.md |
| 10-27 | S0c/S0d | spectrum-tracking / leave-kappa-out test calibrated on toys | REPORT_S0bcd_2026-10-27.md |
| 10-27 | S2 EqR spectra (4 puzzles + replicates) | 7-11 positive exponents (rough allowed); registered NOT-SKEW, noise-limited; 814 reproducibly different | REPORT_S2_eqr_spectrum_2026-10-27.md, POSTHOC_replicate.md |
| 10-27 | v7 GPU kernel | ready, NOT pushed: most likely outcome NOT-SKEW, which does not settle the question | (GPU kernel prepared, not run, not included) |
| 10-27 | S0f long-trajectory CLVs | contamination fixed (3e-12); blind toy lambda_x recovered (0.406 vs 0.405) | REPORT_S0f_2026-10-27.md |
| 10-28 | S3a exact JVP | forward-AD JVPs through EqR work (call m.inner.deep_recursion; the inner forward detaches) | s3a_jvp.py |
| 10-28 | S3 feasibility (1982) | FAILED at F1: precision-survival conflict (the float64 window needs fast kappa, stagger needs slow kappa) | PREREG_S3_feasibility.md |
| 10-28 | S4 survival-regression spectroscopy (toys) | FAILED: survival-stretch correlations are geometric confounds (filam even flips sign) | REPORT_S4_2026-10-28.md |
| 10-28 | A: hidden roughness (theory + toys) | degeneracy argument; rounding law floor ~ eps^h (3/3); contamination law ~ c^(h/(1-h)) (2/3, third favourable); EqR undecidable in float64, puzzle 80 ~2.4 decades with double-double | REPORT_A_hidden_roughness_2026-10-28.md |
| 10-28 | A3 islands vs stripes (toys) | FAILED: skew roughness is effectively 1-D at fine scales, so both classes look like stripes | REPORT_A3_islands_2026-10-28.md |
| 10-28 | A4 architectural coupling (EqR, exact JVPs) | no invariant token group (all leak >= 0.53); contested cells partly self-contained (0.53-0.61 vs random 0.73-0.80) | REPORT_A4_coupling_2026-10-28.md |
| 10-28 | S5 escape-triggered averages (toys) | FAILED: generic observables relax at transfer-operator rates; a filamentary toy faked a rough rate (~kappa) | REPORT_S5_eta_2026-10-28.md |
| 10-28 | S0g gate for option B (toys with feedback eps) | FAILED: feedback 1e-3 flips the probe to FILAM (cross slope 2.46 -> 1.01); B stopped before engineering | REPORT_S0g_gateB_2026-10-28.md |
| 10-28 | S0h true kick-free orbits (80 digits, toys with feedback) | RESCUED: 9/9 correct ROUGH, lambda_x within 25% in 8/9; complete cost law -> EqR needs ~450-850-loop genuine orbits in ~460-850 digits | REPORT_S0h_true_orbit_2026-10-28.md |

Current position (10-28): every route is blocked for EqR. Generic observables are degenerate or confounded; the
exact-direction probe needs precision (laws, REPORT_A) AND, with any feedback, near-exact long saddle orbits (S0g).
