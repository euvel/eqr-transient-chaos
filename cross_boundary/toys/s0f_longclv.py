"""S0f: does the DIRECT lambda_x measurement work once CLVs come from LONG stagger trajectories? (S0b toys, pure numpy)

S0b failed for two reasons: float64 noise (fixed by the ~16*alpha-decade fit window, Amendment 1 of S0b) and CLV
contamination from a backward iteration over only ~15 steps. Here every trajectory is kept on the saddle by
stagger-and-step for L = 200 steps, and the CLVs at the middle come from 100 backward steps.
Probe: from the middle state z, follow its UNKICKED future to 8 steps before it settles, carrying the CLVs forward with
the Jacobian (covariant); then run direction-resolved PIM along each unstable CLV (S0b Amendment 1 window and
adaptive half-width) and along a random direction.
lambda_x_hat = 1 / max_k median slope_k over unstable CLVs; ROUGH iff lambda_x_hat / lambda_1 < 0.7.

Expectations (written before the first run): ROUGH -> lambda_x_hat ~ 0.405 (within 20%), verdict ROUGH;
FILAM -> lambda_x_hat ~ 1.10 (within 20%), verdict FILAM. CLV contamination at the probe point < 1e-4.

Amendment 1 (2026-10-27, after the first run: contamination 1.3e-14 as hoped, but every ROUGH probe was empty because
stagger lost the trajectory, y ~ 1e12): the rescue came too late (look-ahead 12 -> already ~0.08 off the graph) with
kicks too small (<= 1e-3). Stagger now uses look-ahead 20, kicks in [1e-8, 1e-2], backtracking by 6 steps with a forced
kick, and asserts the trajectory never settles. The estimator and expectations are unchanged.
"""
import json

import numpy as np

import s0b_blind as b

rng = np.random.default_rng(5)
L, MID, TAU, NTRAJ, K = 200, 100, 12, 8, 5


def alive(U, toy, tau=TAU):
    return b.settle_time(U, toy) > tau


def stagger(toy, tau=20, adv=6):
    U = b.random_starts(20000, toy)
    traj = [U[np.argmax(b.settle_time(U, toy))]]
    must_kick = False
    while len(traj) < L:
        u = traj[-1]
        if must_kick or not alive(u[None], toy, tau)[0]:
            ok_any = False
            for _ in range(20):
                d = 10 ** rng.uniform(-8, -2, 32)
                w = rng.standard_normal((32, b.N))
                w /= np.linalg.norm(w, axis=1, keepdims=True)
                C = u + d[:, None] * np.linalg.norm(u) * w
                ok = alive(C, toy, tau)
                if ok.any():
                    traj[-1] = C[np.argmin(np.where(ok, d, np.inf))]
                    ok_any = True
                    break
            if not ok_any:  # backtrack and force a kick from the earlier state
                assert len(traj) > adv + 1, "stagger cannot recover"
                del traj[-adv:]
                must_kick = True
                continue
            must_kick = False
        for _ in range(adv):
            traj.append(b.F(traj[-1], toy))
    traj = traj[:L]
    assert all(b.settle_time(x[None], toy)[0] > 0 for x in traj), "trajectory settled"
    return traj


def clvs(traj, toy):
    Q = np.linalg.qr(rng.standard_normal((b.N, K)))[0]
    Qs, Rs, loc = [Q], [], []
    for x in traj[:-1]:
        Q, R = np.linalg.qr(b.J_ext(x, toy) @ Q)
        sg = np.sign(np.diag(R))
        Q, R = Q * sg, R * sg[:, None]
        Qs.append(Q)
        Rs.append(R)
        loc.append(np.log(np.abs(np.diag(R))))
    C = np.eye(K)
    for t in range(len(Rs) - 1, MID - 1, -1):
        C = np.linalg.solve(Rs[t], C)
        C /= np.linalg.norm(C, axis=0)
    return np.array(loc)[6:].mean(0), Qs[MID] @ C


def contamination(V, toy, z):
    """Internal components: for ROUGH, CLV3 (the y direction) should have no x1/x2 part."""
    Vi = b.R.T @ V
    return float(np.abs(Vi[:2, 2]).max())


def run(toy, kappa):
    lams, slopes, rnd, cont = [], [], [], []
    for _ in range(NTRAJ):
        traj = stagger(toy)
        lam, V = clvs(traj, toy)
        z = traj[MID]
        Tz = int(b.settle_time(z[None], toy)[0])
        for _ in range(max(0, Tz - 8)):  # unkicked future of z, CLVs carried forward
            V = b.J_ext(z, toy) @ V
            V /= np.linalg.norm(V, axis=0)
            z = b.F(z, toy)
        cont.append(contamination(V, toy, z))
        wmin = max(np.exp(-0.8 * 36.8 * kappa / lam[0]), 1e-11)
        lams.append(lam)
        slopes.append([b.lifetime_slope(z, V[:, k], toy, wmin) if lam[k] > 0.05 else np.nan for k in range(K)])
        v = rng.standard_normal(b.N)
        rnd.append(b.lifetime_slope(z, v / np.linalg.norm(v), toy, wmin))
    lam = np.median(np.array(lams), 0)
    S = np.array(slopes)
    med = np.array([np.nanmedian(S[:, k]) if np.isfinite(S[:, k]).any() else np.nan for k in range(K)])
    lx = 1 / np.nanmax(med[lam > 0.05])
    return {"kappa": round(kappa, 3), "spectrum": lam.round(3).tolist(), "clv_slopes": np.round(med, 3).tolist(),
            "nan_frac": np.isnan(S).mean(0).round(2).tolist(), "random_slope": round(float(np.nanmedian(rnd)), 3),
            "contamination_clv3_median": float(np.median(cont)) if toy == "rough" else None,
            "lambda_x_hat": round(float(lx), 3), "ratio": round(float(lx / lam[0]), 3),
            "verdict": "ROUGH" if lx / lam[0] < 0.7 else "FILAM"}


if __name__ == "__main__":
    res = {}
    for toy in ("rough", "filam"):
        T = b.settle_time(b.random_starts(100_000, toy), toy)
        S = np.array([(T > j).mean() for j in range(b.CAP)])
        m = (S <= 0.9) & (S >= 1e-3)
        kap = float(-np.polyfit(np.arange(b.CAP)[m], np.log(S[m]), 1)[0])
        res[toy] = run(toy, kap)
        print(toy, res[toy], flush=True)
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
