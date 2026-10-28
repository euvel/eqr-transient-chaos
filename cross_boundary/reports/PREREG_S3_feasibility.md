# S3: feasibility smoke for the DIRECT lambda_x measurement on EqR (laptop, written 2026-10-28 before any S3 data)
Puzzle 1982 (kappa 0.477, lambda_1 ~2.35 from the maps, alpha ~0.2 -> noise window ~1.3-1.6 decades).
Pipeline: s2b_stagger (Amendment 2 settings) -> s3c_clv_ad (exact forward-AD JVPs, K = 12, CLVs at MID with >= 45
backward loops) -> s3d_probe (PIM along CLV_1, the CLV with lambda nearest kappa, the last positive CLV, and a random
direction). S3a showed AD JVPs are exact (central differences converge to it as eps^2).

Feasibility criteria (all must hold for a GPU run to be worth designing):
 F1 stagger keeps 1982 chaotic for >= 120 loops in <= 3 laptop chunks (kick success, few backtracks, no failed rescues)
 F2 CLV convergence: |V(full) - V(end-15)| <= 1e-6 for the probed CLVs
 F3 each probe gives >= 3 levels inside the noise window with T variation (a fittable slope)
 F4 CLV_1 slope within 40% of 1/lambda_1 (sanity: the probe works on EqR)

Reading (exploratory, one probe point; NOT a verdict): a slope along some unstable CLV_k (k >= 2) that is clearly larger
than 1/lambda_1 and close to 1/lambda_k would be a rough-type signal worth a GPU run; all slopes ~ 1/lambda_1 would be
filamentary-type. Either way, the GPU run would be designed and pre-registered separately.

Amendment S3-1 (after the first stagger chunk): with look-ahead 16, both rescue rounds for 1982 failed (0/48 candidates
alive; ~7 min per round; 13 loops in 14 min). Retry with look-ahead 8, advance 4 (env STAGGER_TAU/ADV), same kicks.
F1 is judged on this retry.

## Outcome (2026-10-28): F1 FAILED -> smoke stopped before S3c/S3d
- Look-ahead 16: 13 loops in 14 min, 2/2 rescue rounds failed (0/48 candidates alive).
- Look-ahead 8: 1 backtrack + 2 failed rescues; the trajectory snapped (relative step 0.013 at the end); 5 usable loops.
- Diagnosis, a precision-survival conflict: the float64 window for the cross direction is ~13*alpha decades
  (0.8 * 16 * kappa/lambda_1), so it needs FAST puzzles; stagger-and-step needs candidates to survive the look-ahead
  (~exp(-kappa * tau)), so it needs SLOW puzzles. Where stagger works (kappa <= 0.25: 206, 814, 225, 80) the window is
  <= ~1 decade (80: ~0.5-1); where the window is usable (kappa >= 0.48) stagger fails.
