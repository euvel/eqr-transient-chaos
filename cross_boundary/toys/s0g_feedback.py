"""S0g (gate for option B): does the long-trajectory CLV probe (S0f) survive WEAK FEEDBACK from the decision into the churn?
S0f's rough toy is exactly skew (block-triangular Jacobian), so its cross direction is the same at every point and
stagger kicks are harmless. EqR has no invariant subspace (A4). Here x1' gets + eps * tanh(y): the cross direction now
varies along the orbit and depends on its future, so kicks may pollute the computed CLVs.
Prediction (written before the run): with eps = 0 the S0f result repeats (lambda_x_hat ~ 0.405, ROUGH). If at
eps = 1e-3 or 1e-2 the verdict flips to FILAM or lambda_x_hat moves > 30% from ~0.41, B as designed (CLVs along kicked
stagger trajectories) is not viable for EqR without a different trajectory source.
"""
import json
import numpy as np
import s0b_blind as b
import s0f_longclv as f

base_step, base_jac = b.step, b.jac


def make(eps):
    def step(P, toy):
        out = base_step(P, toy)
        out[:, 0] = (out[:, 0] + eps * np.tanh(P[:, 2])) % 1.0
        return out

    def jac(p, toy):
        J = base_jac(p, toy)
        J[0, 2] += eps / np.cosh(p[2]) ** 2
        return J
    return step, jac


res = {}
for eps in (0.0, 1e-3, 1e-2):
    b.step, b.jac = make(eps)
    T = b.settle_time(b.random_starts(100_000, "rough"), "rough")
    S = np.array([(T > j).mean() for j in range(b.CAP)])
    m = (S <= 0.9) & (S >= 1e-3)
    kap = float(-np.polyfit(np.arange(b.CAP)[m], np.log(S[m]), 1)[0])
    r = f.run("rough", kap)
    r.pop("contamination_clv3_median", None)
    res[str(eps)] = r
    print(f"eps {eps}: kappa {kap:.3f} spectrum {r['spectrum'][:3]} slopes {r['clv_slopes'][:3]} "
          f"lambda_x_hat {r['lambda_x_hat']} verdict {r['verdict']}", flush=True)
json.dump(res, open("s0g_feedback.json", "w"), indent=1, default=float)
