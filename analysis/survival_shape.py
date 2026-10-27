"""Is EqR settle-time survival exponential, or super-exponential ("doubly transient")?

Criteria fixed BEFORE running (written in this docstring first):
Data: every existing settle-time map with a long cap (v3b + v4-p0 main maps: 1600 points, cap 150; v2 HARD_p1 + P3: cap 120).
T = T_dec; T == cap-1 is censored (still changing at the end).
Discrete hazard at loop k: n_k = #(T >= k), d_k = #(T == k), k < cap-1.
Window: k0 = first k with S(k) <= 0.9 (same as the kappa fit), k1 = last k with n_k >= 20. Need k1 - k0 >= 6.
Model: continuous-time rate r_k = exp(a + g (k - kmid)), P(event at k | at risk) = 1 - exp(-r_k)  (cloglog), ML fit.
g = 0 is exponential survival; doubly transient chaos predicts g > 0 (settling rate growing exponentially in time).
Per puzzle, with R = exp(g (k1 - k0)) = rate change across the window and the 95% Wald CI of g:
  EXPONENTIAL      CI of g contains 0, or 1/1.5 < R < 1.5
  SUPER-EXP        g > 0 significant AND R >= 1.5
  SUB-EXP (sticky) g < 0 significant AND R <= 1/1.5
Overall claim "not doubly transient within the window" holds if no puzzle is SUPER-EXP, or SUPER-EXP puzzles are a
minority AND the pooled g (inverse-variance) has CI containing 0 or below 0.
Robustness (reported): same fit on the late half of the window only; T_tau instead of T_dec.
"""
import json
import numpy as np
from scipy.optimize import minimize

from common import RES

maps = {}
for f in sorted(RES.glob("maps_v3b/main_r*.npz")) + sorted(RES.glob("maps_v4/main_r*.npz")):
    maps["r" + f.stem[6:]] = (f, 150)
for nm in ("HARD_p1", "P3_p0", "P3_p1"):
    maps["v2_" + nm] = (RES / f"maps_v2/{nm}.npz", 120)


def fit(T, cap, lo=None, hi=None):
    T = np.asarray(T)
    ks = np.arange(1, cap - 1)
    n = np.array([(T >= k).sum() for k in ks]); d = np.array([(T == k).sum() for k in ks])
    S = np.array([(T > k).mean() for k in ks])
    if not (S <= 0.9).any(): return None
    k0 = ks[np.argmax(S <= 0.9)]
    ok = np.where(n >= 20)[0]
    k1 = ks[ok[-1]] if len(ok) else k0
    if lo is not None: k0 = max(k0, lo)
    if hi is not None: k1 = min(k1, hi)
    if k1 - k0 < 6: return {"k0": int(k0), "k1": int(k1), "skip": "window < 6"}
    m = (ks >= k0) & (ks <= k1)
    k, nn, dd = ks[m], n[m], d[m]; km = (k0 + k1) / 2
    def nll(p):
        r = np.exp(np.clip(p[0] + p[1] * (k - km), -30, 5)); q = 1 - np.exp(-r)
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -(dd * np.log(q) + (nn - dd) * np.log(1 - q)).sum()
    h0 = max(dd.sum() / nn.sum(), 1e-4)
    res = minimize(nll, [np.log(-np.log(1 - h0)), 0.0], method="BFGS")
    # numerical Hessian for SE
    eps = 1e-4; H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i, e_j = np.eye(2)[i] * eps, np.eye(2)[j] * eps
            H[i, j] = (nll(res.x + e_i + e_j) - nll(res.x + e_i - e_j) - nll(res.x - e_i + e_j) + nll(res.x - e_i - e_j)) / (4 * eps ** 2)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    g, sg = res.x[1], se[1]
    R = float(np.exp(g * (k1 - k0)))
    lo95, hi95 = g - 1.96 * sg, g + 1.96 * sg
    if lo95 <= 0 <= hi95 or 1 / 1.5 < R < 1.5: v = "EXPONENTIAL"
    elif g > 0: v = "SUPER-EXP"
    else: v = "SUB-EXP"
    # nll of g=0 for LR
    r0 = minimize(lambda p: nll([p[0], 0.0]), [res.x[0]], method="BFGS")
    return {"k0": int(k0), "k1": int(k1), "n_events": int(dd.sum()), "kappa_mid": float(np.exp(res.x[0])),
            "g": float(g), "g_se": float(sg), "R": R, "LR": float(2 * (r0.fun - res.fun)), "verdict": v,
            "S_k0": float(S[ks == k0][0]), "S_k1": float(S[ks == k1][0])}


out = {}
print(f"{'map':>12} {'win':>9} {'ev':>5} {'kappa':>6} {'g (/loop)':>17} {'R':>5} {'LR':>6}  verdict   | late-half | T_tau")
gs, ws = [], []
for name, (f, cap) in maps.items():
    z = np.load(f)
    r = fit(z["T_dec"], cap)
    if r is None or "skip" in r:
        print(f"{name:>12} skipped ({r})"); out[name] = r; continue
    late = fit(z["T_dec"], cap, lo=(r["k0"] + r["k1"]) // 2)
    tau = fit(z["T_tau"], cap)
    out[name] = {"T_dec": r, "late_half": late, "T_tau": tau}
    gs.append(r["g"]); ws.append(1 / r["g_se"] ** 2)
    lv = late["verdict"] if late and "verdict" in late else "-"
    tv = tau["verdict"] if tau and "verdict" in tau else "-"
    print(f"{name:>12} {r['k0']:>3}-{r['k1']:<4} {r['n_events']:>5} {r['kappa_mid']:6.3f} {r['g']:+.4f} ± {1.96*r['g_se']:.4f} {r['R']:5.2f} {r['LR']:6.1f}  {r['verdict']:<11}| {lv:<11}| {tv}")
gs, ws = np.array(gs), np.array(ws)
gp = float((gs * ws).sum() / ws.sum()); sp = float(1 / np.sqrt(ws.sum()))
cnt = {v: sum(1 for x in out.values() if x and "T_dec" in x and x["T_dec"]["verdict"] == v) for v in ("EXPONENTIAL", "SUPER-EXP", "SUB-EXP")}
print("\ncounts:", cnt)
print(f"pooled g = {gp:+.4f} ± {1.96*sp:.4f} /loop (95%)")
out["_pooled"] = {"g": gp, "g_se": sp, "counts": cnt}
lg = [x["late_half"] for k, x in out.items() if not k.startswith("_") and x and "late_half" in x and x["late_half"] and "g" in x["late_half"]]
g2 = np.array([x["g"] for x in lg]); w2 = np.array([1 / x["g_se"] ** 2 for x in lg])
gl, sl = float((g2 * w2).sum() / w2.sum()), float(1 / np.sqrt(w2.sum()))
print(f"late half: {sum(x['verdict'] == 'EXPONENTIAL' for x in lg)}/{len(lg)} EXPONENTIAL; pooled g = {gl:+.4f} ± {1.96*sl:.4f} /loop;"
      f" rate change over 50 loops x{np.exp(gl*50):.2f} (95% upper bound x{np.exp((gl+1.96*sl)*50):.2f})")
