# S5: escape-triggered averages (ETA) on toys (numpy, 2026-10-28): FAILED as a quantitative discriminator
Script `s5_eta_toys.py` (prediction in the docstring), raw `s5_eta_toys.json`. Runtime 14 s.

| toy | kappa | lambda_1 | events | O_esc (escape coordinate) | O_gen (generic mix) |
|---|---|---|---|---|---|
| ROUGH a=1.2 | 0.179 | 1.10 | 9043 | gradual, rate 0.246 (+38%) | gradual, rate 0.386 (+115%) |
| ROUGH a=1.5 | 0.399 | 1.10 | 20 | gradual, 0.448 (+12%, n tiny) | abrupt |
| FILAM mu=4.2 | 0.201 | 1.39 | 2165 | abrupt (correct) | **gradual, rate 0.213 ~ kappa (a fake rough signal)** |
| FILAM mu=4.6 | 0.404 | 1.89 | 11 | abrupt | abrupt |

Prediction (ROUGH: rate within 30% of kappa; FILAM: abrupt or >= 0.8 lambda_1) fails.
Why: an escape-triggered average measures how the distribution of states CONDITIONED on escaping in tau loops relaxes to
the saddle's average. That relaxation is set by the open transfer operator's spectrum and by each observable's
nonlinearity (|y| responds to the drift d only at second order), not by lambda_x alone. A generic observable in a
FILAMENTARY toy relaxed at a rate close to kappa, the exact signature we would have read as "rough". Only an observable
linear in the true escape coordinate gives lambda_x, and that coordinate is what we cannot find. Another face of the
confounding seen in S4.
