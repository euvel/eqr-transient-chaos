"""S0c: calibrate the SPECTRUM-TRACKING test (needs exponents only, no directions) on toy families (pure numpy).

Rationale (S0b): direction-resolved probes are blocked in high dimension by float64 noise and CLV contamination.
Exponents themselves come robustly from forward QR. In a skew-rough system the decision instability is itself a
Lyapunov exponent and sets the escape rate (kappa = lambda_x), while the churn exponents do not depend on it. In a
filamentary system kappa = lambda_1 (1 - d_1) comes from the geometry of the saddle and is not an exponent.

Families (30-D, hidden basis, same churn as S0b: exponents ~ln3, ln2):
  ROUGH(a): y' = a y + forcing, settled |y| > 10, a in {1.2, 1.35, 1.5, 1.8, 2.3}   (kappa = ln a varies, exponent ln a)
  FILAM(w): y' = 0.5 y + forcing, settled when x1 in a hole of width w (+ modulation by x2), w = 1 - 1/a for the same a
            (kappa ~ -ln(1 - w), similar range; spectrum fixed)

Test (blind): per member, top-5 spectrum on the 12 longest-lived trajectories and kappa from survival of 50k starts.
  d = min over positive exponents k >= 2 of |lambda_k - kappa|.
  TRACKS iff d <= 0.1 * kappa + 0.03 for every member AND the matched exponent has Spearman >= 0.9 with kappa.

Expectations (written before the first run): ROUGH family TRACKS; FILAM family does NOT (at most a coincidental match
for the member whose kappa lands near ln2 = 0.69).
"""
import json
import numpy as np
from scipy.stats import spearmanr

N, CAP = 30, 150
rng = np.random.default_rng(1)
R, _ = np.linalg.qr(rng.standard_normal((N, N)))
C = rng.uniform(0, 2 * np.pi, 27)
Wy = rng.standard_normal(27)


def make(kind, par):
    a = par if kind == "rough" else 0.5
    w = 1 - 1 / par

    def step(P):
        x1, x2, y, s = P[:, 0], P[:, 1], P[:, 2], P[:, 3:]
        out = np.empty_like(P)
        out[:, 0] = (3 * x1 + 0.1 * np.sin(2 * np.pi * x2)) % 1.0
        out[:, 1] = (2 * x2) % 1.0
        out[:, 2] = a * y + np.cos(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2)
        out[:, 3:] = 0.5 * s + 0.3 * np.sin(2 * np.pi * x1[:, None] + C) + 0.05 * Wy * y[:, None]
        return out

    def settled(P):
        if kind == "rough":
            return np.abs(P[:, 2]) > 10
        c = 0.5 + 0.05 * np.sin(2 * np.pi * P[:, 1])
        return np.abs(P[:, 0] - c) < w / 2

    def jac(p):
        J = np.zeros((N, N))
        J[0, 0], J[0, 1] = 3.0, 0.2 * np.pi * np.cos(2 * np.pi * p[1])
        J[1, 1] = 2.0
        J[2, 0], J[2, 1], J[2, 2] = -2 * np.pi * np.sin(2 * np.pi * p[0]), -np.pi * np.sin(2 * np.pi * p[1]), a
        J[3:, 0] = 0.6 * np.pi * np.cos(2 * np.pi * p[0] + C)
        J[3:, 2] = 0.05 * Wy
        J[3:, 3:] = 0.5 * np.eye(27)
        return R @ J @ R.T  # pipeline only sees the rotated Jacobian

    return step, settled, jac


def member(kind, par, nstart=50_000, ntraj=12, K=5, skip=4):
    step, settled, jac = make(kind, par)
    P = np.column_stack([rng.random(nstart), rng.random(nstart), rng.uniform(-3, 3, nstart),
                         rng.uniform(-1, 1, (nstart, 27))])
    P0 = P.copy()
    T = np.full(nstart, CAP)
    alive = np.ones(nstart, bool)
    for t in range(CAP):
        d = alive & settled(P)
        T[d] = t
        alive &= ~d
        P = step(P)
    S = np.array([(T > j).mean() for j in range(CAP)])
    m = (S <= 0.9) & (S >= 1e-3)
    kappa = float(-np.polyfit(np.arange(CAP)[m], np.log(S[m]), 1)[0])
    lams = []
    for i in np.argsort(T)[::-1][:ntraj]:
        p = P0[i:i + 1]
        Q = np.linalg.qr(rng.standard_normal((N, K)))[0]
        loc = []
        for _ in range(T[i]):
            Q, Rm = np.linalg.qr(jac(p[0]) @ Q)
            loc.append(np.log(np.abs(np.diag(Rm))))
            p = step(p)
        loc = np.array(loc)
        lams.append(loc[skip:len(loc) - skip].mean(0))
    lam = np.median(np.array(lams), 0)
    pos = [l for l in lam[1:] if l > 0.05]
    match = min(pos, key=lambda l: abs(l - kappa)) if pos else float("nan")
    return {"par": par, "kappa": round(kappa, 3), "spectrum": lam.round(3).tolist(), "matched": round(float(match), 3),
            "d": round(abs(match - kappa), 3), "ok": bool(abs(match - kappa) <= 0.1 * kappa + 0.03)}


if __name__ == "__main__":
    res = {}
    for kind in ("rough", "filam"):
        rows = [member(kind, a) for a in (1.2, 1.35, 1.5, 1.8, 2.3)]
        rho = spearmanr([r["kappa"] for r in rows], [r["matched"] for r in rows])[0]
        res[kind] = {"rows": rows, "spearman_matched_vs_kappa": round(float(rho), 3),
                     "TRACKS": bool(all(r["ok"] for r in rows) and rho >= 0.9)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
