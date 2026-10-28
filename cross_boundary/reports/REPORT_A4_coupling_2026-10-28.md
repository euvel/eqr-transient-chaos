# A4: is churn-drives-decision built into EqR's architecture? Token-group coupling (exact JVPs, laptop, 2026-10-28)
Script `a4_coupling.py` (prediction in the docstring), raw `a4_coupling.json`. Runtime 4.4 min.
States on saddle trajectories of 206 and 80 (loops 20 and 40). Leakage = share of the response norm outside the
perturbed token group (0 = the group is an invariant subspace).

| group | size | leakage after 1 loop | after 3 loops | random group of same size (1 / 3 loops) |
|---|---|---|---|---|
| prefix (puzzle embedding) | 16 | 0.97-0.99 | 0.98-1.00 | 0.90-0.94 / 0.84-0.93 |
| contested cells (value changes on the saddle) | 35-38 | **0.53-0.61** | 0.60-0.68 | 0.73-0.80 / 0.76-0.83 |
| settled cells | 43-46 | 0.76-0.85 | 0.73-0.82 | 0.68-0.74 / 0.70-0.79 |
| one contested cell | 1 | 0.99 | 0.97-1.00 | 0.99-1.00 |
| z_H only -> into z_L | - | 0.89-0.93 | - | - |
| z_L only -> into z_H | - | 0.36-0.42 | - | - |

Verdict: **no architectural invariant subspace.** Every group leaks >= 0.53 after one loop (prediction: >= 0.3,
confirmed). The prediction "close to the random baseline" holds only roughly: the contested cells form a partly
self-contained cluster, keeping more of their own perturbation than a random group of the same size (leakage
0.53-0.61 vs 0.73-0.80), while the settled cells leak more. Not exact, so the cross direction cannot be read off the
architecture, and the precision laws of REPORT_A apply in full.
Side observation (descriptive): the contested cells are a dynamically "sticky" cluster, a possible handle for
interpreting where the churn lives.
