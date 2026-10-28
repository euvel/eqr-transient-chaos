"""S0h: does a KICK-FREE true saddle orbit (80 digits) rescue the cross-direction probe under feedback?

Toy (3-D, mpmath): x1' = 3 x1 + 0.1 sin 2pi x2 + eps tanh(y) mod 1, x2' = 2 x2 mod 1,
y' = 1.5 y + cos 2pi x1 + 0.5 cos 2pi x2; settled |y| > 10. Rough, lambda_x = ln 1.5 = 0.405, lambda_1 ~ ln 3.
Pipeline, all at 80 digits: (1) bisection (PIM) along a random line until a starting point lives >= 160 steps:
a genuine orbit, no kicks; (2) Lyapunov QR forward along it, Ginelli backward from step T-8 to M = 60 (gap ~0.29,
~90 backward steps); (3) bisection probe at z_M along CLV1, CLV2, CLV3; lifetime slopes over widths [1e-7, 1e-1].
lambda_x_hat = 1 / max slope. eps in {0, 1e-3, 1e-2}, 3 orbits each.
Prediction (written before the run): lambda_x_hat within 25% of 0.405 (verdict ROUGH) for every eps. Then S0g's failure
came from the kicked pseudo-orbit, and the cost of deciding is set by how long a near-true orbit must be.
Bug fix after the first run (prediction unchanged): the bisection probe was centred on z_M (lifetime ~100), so it
never moved and every slope was empty. Lifetimes are now sampled directly at offsets +-w, w = 1e-1..1e-7.
"""
import json
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 80
TWO_PI = 2 * mp.pi
CAP, TLONG, M = 400, 160, 60
rng = np.random.default_rng(8)


def step(p, eps):
    x1, x2, y = p
    n1 = 3 * x1 + mp.mpf("0.1") * mp.sin(TWO_PI * x2) + eps * mp.tanh(y)
    n2 = 2 * x2
    return [n1 - mp.floor(n1), n2 - mp.floor(n2), mp.mpf("1.5") * y + mp.cos(TWO_PI * x1) + mp.mpf("0.5") * mp.cos(TWO_PI * x2)]


def jac(p, eps):
    x1, x2, y = p
    return mp.matrix([[3, mp.mpf("0.1") * TWO_PI * mp.cos(TWO_PI * x2), eps / mp.cosh(y) ** 2],
                      [0, 2, 0],
                      [-TWO_PI * mp.sin(TWO_PI * x1), -mp.pi * mp.sin(TWO_PI * x2), mp.mpf("1.5")]])


def lifetime(p, eps):
    for t in range(CAP):
        if abs(p[2]) > 10:
            return t
        p = step(p, eps)
    return CAP


def pim(p0, d, eps, half, stop, target=None):
    lo, hi = -half, half
    W, TM, best = [], [], None
    while hi - lo > stop:
        ss = [lo + (hi - lo) * i / 8 for i in range(9)]
        T = [lifetime([p0[k] + s * d[k] for k in range(3)], eps) for s in ss]
        i = int(np.argmax(T))
        W.append(float(hi - lo))
        TM.append(T[i])
        best = [p0[k] + ss[i] * d[k] for k in range(3)]
        if target and T[i] >= target:
            break
        lo, hi = ss[max(i - 1, 0)], ss[min(i + 1, 8)]
    return np.array(W), np.array(TM), best


def unit(v):
    n = mp.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def run(eps, seed):
    r = np.random.default_rng(seed)
    for _ in range(5):  # find a genuine long orbit
        p0 = [mp.mpf(r.random()), mp.mpf(r.random()), mp.mpf(r.uniform(-2, 2))]
        d = unit([mp.mpf(x) for x in r.standard_normal(3)])
        _, TM, z0 = pim(p0, d, eps, mp.mpf("0.5"), mp.mpf(10) ** -75, target=TLONG)
        if TM.max() >= TLONG:
            break
    T = lifetime(z0, eps)
    orbit = [z0]
    for _ in range(T):
        orbit.append(step(orbit[-1], eps))
    Q = mp.eye(3)
    Qs, Rs, loc = [Q], [], []
    for t in range(T - 8):
        Q, R = mp.qr(jac(orbit[t], eps) * Q)
        for k in range(3):  # positive diagonal
            if R[k, k] < 0:
                for i in range(3):
                    Q[i, k] = -Q[i, k]
                for j in range(3):
                    R[k, j] = -R[k, j]
        Qs.append(Q)
        Rs.append(R)
        loc.append([float(mp.log(R[k, k])) for k in range(3)])
    C = mp.eye(3)
    for t in range(len(Rs) - 1, M - 1, -1):
        C = mp.inverse(Rs[t]) * C
        for k in range(3):
            n = mp.sqrt(sum(C[i, k] ** 2 for i in range(3)))
            for i in range(3):
                C[i, k] /= n
    V = Qs[M] * C
    zM = orbit[M]
    slopes = []
    for k in range(3):
        v = [V[i, k] for i in range(3)]
        # z_M lies on the boundary to ~1e-45, so sample the lifetime directly at offsets +-w (bug fix: a bisection
        # centred on z_M never leaves it)
        ws = np.logspace(-1, -7, 13)
        TM = np.array([np.mean([lifetime([zM[i] + sg * mp.mpf(w) * v[i] for i in range(3)], eps) for sg in (1, -1)])
                       for w in ws])
        slopes.append(float(np.polyfit(np.log(1 / ws), TM, 1)[0]) if np.ptp(TM) > 0 else None)
    lam = np.mean(loc[6:], 0)
    good = [s for s in slopes if s]
    lx = 1 / max(good) if good else None
    return {"T_orbit": T, "spectrum": lam.round(3).tolist(), "slopes": slopes, "lambda_x_hat": lx,
            "verdict": None if lx is None else ("ROUGH" if lx / lam[0] < 0.7 else "FILAM")}


if __name__ == "__main__":
    out = {}
    for eps in ("0", "1e-3", "1e-2"):
        out[eps] = [run(mp.mpf(eps), s) for s in (1, 2, 3)]
        for r in out[eps]:
            print(f"eps {eps}: T {r['T_orbit']} spectrum {r['spectrum']} slopes {np.round([x or np.nan for x in r['slopes']], 2).tolist()} "
                  f"lambda_x_hat {r['lambda_x_hat'] and round(r['lambda_x_hat'], 3)} {r['verdict']}", flush=True)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
