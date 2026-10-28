# S0g: gate for option B: does the CLV probe survive weak feedback? (toys, numpy, 2026-10-28): FAILED -> B stopped
Script `s0g_feedback.py` (prediction in the docstring), raw `s0g_feedback.json`. Runtime 33 s.
S0f's rough toy plus feedback x1' += eps * tanh(y) (the decision now feeds the churn weakly, as every EqR token group does, A4).

| eps | kappa | spectrum (top 3) | CLV slopes (1,2,3) | lambda_x_hat | verdict | truth |
|---|---|---|---|---|---|---|
| 0 | 0.399 | 1.10, 0.69, 0.41 | 0.98, 0.28, **2.46** | **0.406** | ROUGH | rough, lambda_x 0.405 |
| 1e-3 | 0.403 | 1.10, 0.69, 0.41 | 0.94, 0.77, **1.01** | 0.99 | **FILAM (wrong)** | rough, lambda_x ~0.405 |
| 1e-2 | 0.396 | 1.10, 0.69, 0.41 | 1.37, 1.16, 1.03 | 0.73 (+80%) | ROUGH | rough, lambda_x ~0.405 |

The spectrum and kappa are unchanged, so the toy stays rough with the same lambda_x, but feedback of only 1e-3 makes
the probe lose the cross-direction signal (2.46 -> 1.01). In S0f the cross direction was an exactly invariant
coordinate, the same at every point, so stagger kicks and finite trajectories were harmless; that was a property of
the toy, not of the method. With feedback the cross direction depends on the orbit's own future, and CLVs computed along
a kicked (pseudo-)orbit no longer match the probe point's true future. Mechanism not fully isolated (kicks vs finite
backward window), but the effect is decisive.

Consequence for option B: double-double arithmetic alone cannot rescue the probe for EqR (which has no invariant
subspace, A4). It would also need near-exact saddle orbits (kicks ~1e-30, look-ahead >= ln(1/kick)/lambda_1 ~ 30 loops)
followed by backward windows of ~ln(1/c)/gap loops (hundreds, for EqR's small spectral gaps), which is not realistic.
B is stopped at its gate. This strengthens the hidden-roughness result: in non-skew systems the cross direction is hidden
not only by precision but by the orbit-dependence of the bundle itself.
