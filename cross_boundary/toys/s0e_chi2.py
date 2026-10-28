"""S0e: calibrate the v7 decision rule (element-wise chi^2 with replicate-calibrated noise) on the S0d dense toys.

Differences from S0d, mirroring the planned EqR run: two LONG replicate saddle trajectories per member, made by
stagger-and-step (look-ahead 12, kicks 1e-8..1e-3 relative); block-bootstrap SEs; no counting of positive exponents.

Rule (identical to PREREG_v7, with M = 7 here and 12 for EqR):
  per member i, replicates A, B: lambda = (A+B)/2, SE^2 = (SE_A^2 + SE_B^2)/4
  replicate check chi2_rep,i = sum_{k<=M} (A_k-B_k)^2/(SE_A^2+SE_B^2);  inflation f = max(1, sqrt(median_i chi2_rep,i / M))
  full:    chi2_full(i,j) = sum_{k<=M} (l_ik - l_jk)^2 / (f^2 (SE_ik^2 + SE_jk^2))
  removed: drop each member's exponent nearest kappa_i among the first M+1, compare the first M
  AGREE (full / removed) iff every pair has chi2 < chi2_{M}(0.99)
  ROUGH-SKEW iff removed AGREE and full not; FILAM-CONSISTENT iff full AGREE and removed not;
  UNRESOLVED iff both agree; NOT-SKEW iff neither.
Expectations (written before the first run): ROUGH family -> ROUGH-SKEW, FILAM family -> FILAM-CONSISTENT.
"""
import json

import numpy as np
from scipy.stats import chi2

import s0d_dense as T

rng = np.random.default_rng(11)
M, K, L, TAU = 7, 10, 300, 12


def stagger(kind, par):
    step, settled, jac = T.make(kind, par)

    def lifetime(P, n):
        alive = np.ones(len(P), bool)
        for _ in range(n):
            alive &= ~settled(P)
            P = step(P)
        return alive

    P = np.column_stack([rng.random((4000, T.M)), rng.uniform(-3, 3, 4000), rng.uniform(-1, 1, (4000, T.NS))])
    p = P[np.argmax(lifetime(P.copy(), 20) * rng.random(4000))][None]  # a start alive at 20 steps
    traj = []
    while len(traj) < L:
        if not lifetime(p.copy(), TAU)[0]:
            for _ in range(50):
                d = 10 ** rng.uniform(-8, -3, 32)
                w = rng.standard_normal((32, T.N))
                w /= np.linalg.norm(w, axis=1, keepdims=True)
                C = p + d[:, None] * np.linalg.norm(p) * w
                ok = lifetime(C.copy(), TAU)
                if ok.any():
                    p = C[np.argmin(np.where(ok, d, np.inf))][None]
                    break
        traj.append(p[0].copy())
        p = step(p)
    Q = np.linalg.qr(rng.standard_normal((T.N, K)))[0]
    loc = []
    for x in traj:
        Q, Rm = np.linalg.qr(jac(x) @ Q)
        loc.append(np.log(np.abs(np.diag(Rm))))
    return np.array(loc)


def est(loc, start=6, block=8, nboot=1000):
    X = loc[start:]
    nb = len(X) // block
    b = np.array([X[i * block:(i + 1) * block].mean(0) for i in range(nb)])
    return X.mean(0), np.array([b[rng.integers(0, nb, nb)].mean(0) for _ in range(nboot)]).std(0)


def kappa_of(kind, par, n=40_000):
    step, settled, _ = T.make(kind, par)
    P = np.column_stack([rng.random((n, T.M)), rng.uniform(-3, 3, n), rng.uniform(-1, 1, (n, T.NS))])
    Tt = np.full(n, T.CAP)
    alive = np.ones(n, bool)
    for t in range(T.CAP):
        dd = alive & settled(P)
        Tt[dd] = t
        alive &= ~dd
        if not alive.any():
            break
        P = step(P)
    S = np.array([(Tt > j).mean() for j in range(T.CAP)])
    m = (S <= 0.9) & (S >= 1e-3)
    return float(-np.polyfit(np.arange(T.CAP)[m], np.log(S[m]), 1)[0])


def classify(members, M=M, q=0.99):
    lam, se, rep = {}, {}, []
    for k, v in members.items():
        (a, sa), (b, sb) = v["A"], v["B"]
        lam[k], se[k] = (a + b) / 2, np.sqrt(sa ** 2 + sb ** 2) / 2
        rep.append(float(np.sum((a[:M] - b[:M]) ** 2 / (sa[:M] ** 2 + sb[:M] ** 2))))
    f = max(1.0, float(np.sqrt(np.median(rep) / M)))
    rem = {}
    for k, v in members.items():
        j = int(np.argmin(np.abs(lam[k][:M + 1] - v["kappa"])))
        rem[k] = (np.delete(lam[k][:M + 1], j)[:M], np.delete(se[k][:M + 1], j)[:M])
    crit = chi2.ppf(q, M)
    keys = list(members)

    def worst(get):
        w = 0.0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (x, sx), (y, sy) = get(keys[i]), get(keys[j])
                w = max(w, float(np.sum((x - y) ** 2 / (f ** 2 * (sx ** 2 + sy ** 2)))))
        return w

    wf = worst(lambda k: (lam[k][:M], se[k][:M]))
    wr = worst(lambda k: rem[k])
    full, removed = wf < crit, wr < crit
    verdict = {(False, True): "ROUGH-SKEW", (True, False): "FILAM-CONSISTENT", (True, True): "UNRESOLVED",
               (False, False): "NOT-SKEW"}[(full, removed)]
    return {"f_inflation": round(f, 2), "chi2_rep": [round(r, 1) for r in rep], "crit": round(float(crit), 1),
            "worst_chi2_full": round(wf, 1), "worst_chi2_removed": round(wr, 1), "verdict": verdict}


if __name__ == "__main__":
    res = {}
    for kind, pars in (("rough", (1.15, 1.3, 1.5, 1.8)), ("filam", (0.1, 0.2, 0.3, 0.45))):
        mem = {}
        for p in pars:
            mem[p] = {"kappa": kappa_of(kind, p), "A": est(stagger(kind, p)), "B": est(stagger(kind, p))}
        res[kind] = {"members": {str(p): {"kappa": round(v["kappa"], 3), "lam": ((v["A"][0] + v["B"][0]) / 2)[:M + 1]
                                          .round(3).tolist()} for p, v in mem.items()}, **classify(mem)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
