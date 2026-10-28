"""S0d: the LEAVE-KAPPA-OUT spectrum test on DENSE-spectrum toys (pure numpy).

S0c's tracking test works when the spectrum is sparse, but in a dense spectrum some exponent sits near any kappa by
chance. The stricter claims compared across a family that shares the churn and varies kappa:
  ROUGH-SKEW : spectrum_i = (fixed churn set) + one exponent at kappa_i
  FILAM      : spectrum_i = the same set for every i (kappa is not an exponent)
  otherwise  : NOT-SKEW (inconclusive about rough vs filamentary)

Churn: an 8-D hyperbolic toral automorphism (integer, det 1) plus a weak smooth nonlinearity -> several positive
exponents spread over (0, 2). 15 stable directions. Hidden random basis (24-D).
  ROUGH(a): y' = a y + sum_j c_j cos(2 pi x_j), settled |y| > 20; a in {1.15, 1.3, 1.5, 1.8, 2.2, 2.7}.
  FILAM(w): y' = 0.5 y + same forcing; settled when x_1 in a hole of width w in {.1,.2,.3,.45,.6,.75}.

Statistic (blind; positive exponents only, threshold 0.03):
  m_i  = exponent nearest kappa_i; d_i = |m_i - kappa_i|
  A_i  = positive spectrum with m_i removed;  res_out = max over pairs of max |sort(A_i) - sort(A_j)| (inf if lengths differ)
  res_all = same on the full positive spectra
  ROUGH-SKEW iff all d_i <= 0.1 kappa_i + 0.03 and res_out <= 0.08
  FILAM      iff res_all <= 0.08 (count and values unchanged as kappa varies)
  (both cannot hold: in ROUGH the full spectra differ by the moving exponent)

Expectations (written before the first run): ROUGH family -> ROUGH-SKEW; FILAM family -> FILAM.

Amendment 1 (2026-10-27, after the first run: both families NOT-SKEW): members whose longest trajectory is short
(T_max 6-22) give biased finite-time spectra; a verdict-independent sign is that the known stable exponents (-0.69)
drift to -0.36..-0.6 in those members in BOTH families. The test now uses only members with T_max >= 30 (at least 3
required). The same rule will apply to EqR (slow puzzles only). Expectations unchanged.
"""
import json
import numpy as np

M, NS = 8, 15
N = M + 1 + NS
CAP = 200
rng = np.random.default_rng(3)


def unimodular(m, rng, shears=14):
    A = np.eye(m, dtype=np.int64)
    for _ in range(shears):
        i, j = rng.choice(m, 2, replace=False)
        E = np.eye(m, dtype=np.int64)
        E[i, j] = rng.choice([-1, 1])
        A = E @ A
    return A


# pick a churn with several well-spread positive exponents
for _ in range(500):
    A = unimodular(M, rng)
    ev = np.sort(np.log(np.abs(np.linalg.eigvals(A.astype(float)))))[::-1]
    if (ev > 0.1).sum() >= 4 and ev[0] < 2.5 and np.min(np.abs(ev)) > 0.05:
        break
CHURN_EXPONENTS = ev
R, _ = np.linalg.qr(rng.standard_normal((N, N)))
cy = rng.uniform(-1, 1, M)
cy /= np.abs(cy).sum()
Cs = rng.uniform(0, 2 * np.pi, NS)
Ws = rng.standard_normal(NS)
PERM = rng.permutation(M)
NL = 0.02


def make(kind, par):
    a = par if kind == "rough" else 0.5

    def step(P):
        x, y, s = P[:, :M], P[:, M], P[:, M + 1:]
        out = np.empty_like(P)
        out[:, :M] = (x @ A.T + NL * np.sin(2 * np.pi * x[:, PERM])) % 1.0
        out[:, M] = a * y + np.cos(2 * np.pi * x) @ cy
        out[:, M + 1:] = 0.5 * s + 0.3 * np.sin(2 * np.pi * x[:, :1] + Cs) + 0.05 * Ws * y[:, None]
        return out

    def settled(P):
        if kind == "rough":
            return np.abs(P[:, M]) > 20
        return np.abs(P[:, 0] - 0.5) < par / 2

    def jac(p):
        x = p[:M]
        J = np.zeros((N, N))
        J[:M, :M] = A.astype(float)
        J[np.arange(M), PERM] += NL * 2 * np.pi * np.cos(2 * np.pi * x[PERM])
        J[M, :M] = -2 * np.pi * np.sin(2 * np.pi * x) * cy
        J[M, M] = a
        J[M + 1:, 0] = 0.6 * np.pi * np.cos(2 * np.pi * x[0] + Cs)
        J[M + 1:, M] = 0.05 * Ws
        J[M + 1:, M + 1:] = 0.5 * np.eye(NS)
        return R @ J @ R.T

    return step, settled, jac


def member(kind, par, nstart=40_000, ntraj=10, K=10, skip=4):
    step, settled, jac = make(kind, par)
    P = np.column_stack([rng.random((nstart, M)), rng.uniform(-3, 3, nstart), rng.uniform(-1, 1, (nstart, NS))])
    P0 = P.copy()
    T = np.full(nstart, CAP)
    alive = np.ones(nstart, bool)
    for t in range(CAP):
        d = alive & settled(P)
        T[d] = t
        alive &= ~d
        if not alive.any():
            break
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
        sk = min(skip, len(loc) // 4)  # short trajectories in fast members
        lams.append(loc[sk:len(loc) - sk].mean(0))
    lam = np.median(np.array(lams), 0)
    return {"par": par, "kappa": round(kappa, 3), "T_max": int(T.max()), "spectrum": lam.round(3).tolist()}


def classify(rows, thr=0.03, tol=0.08, tmin=30):
    rows = [r for r in rows if r["T_max"] >= tmin]  # Amendment 1
    if len(rows) < 3:
        return {"n_used": len(rows), "verdict": "TOO-FEW"}
    pos = [np.array(sorted([l for l in r["spectrum"] if l > thr], reverse=True)) for r in rows]
    outs, ds = [], []
    for r, p in zip(rows, pos):
        k = int(np.argmin(np.abs(p - r["kappa"])))
        ds.append(abs(p[k] - r["kappa"]) <= 0.1 * r["kappa"] + 0.03)
        outs.append(np.delete(p, k))

    def res(L):
        worst = 0.0
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                if len(L[i]) != len(L[j]):
                    return float("inf")
                worst = max(worst, float(np.max(np.abs(L[i] - L[j]))) if len(L[i]) else 0.0)
        return worst

    r_out, r_all = res(outs), res(pos)
    verdict = "ROUGH-SKEW" if all(ds) and r_out <= tol else ("FILAM" if r_all <= tol else "NOT-SKEW")
    return {"n_used": len(rows), "res_out": r_out, "res_all": r_all, "all_d_ok": bool(all(ds)), "verdict": verdict}


if __name__ == "__main__":
    res = {"churn_exponents_true": CHURN_EXPONENTS.round(3).tolist()}
    for kind, pars in (("rough", (1.15, 1.3, 1.5, 1.8, 2.2, 2.7)), ("filam", (0.1, 0.2, 0.3, 0.45, 0.6, 0.75))):
        rows = [member(kind, p) for p in pars]
        res[kind] = {"rows": rows, **classify(rows)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
