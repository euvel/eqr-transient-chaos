"""A1: the precision law for resolving a rough boundary (2-D Weierstrass-type toy in a rotated basis).

Toy: x' = 3x mod 1, y' = a y + cos(2 pi x); settled when |y| > 10. Repeller = graph y = W(x), Hoelder exponent
h = ln a / ln 3 = lambda_x / lambda_1. The map is computed in a basis rotated by 0.7 rad (u = R p), so every arithmetic
step rounds both coordinates and rounding noise reaches the churn coordinate x, as in a real model.
Probe: bisection (PIM, 9 points/level) along the EXACT cross direction R e_y through an exact graph point (W from its
series at 150 digits). Ideal lifetime growth: T = T0 + ln(1/width)/lambda_x. The floor delta_floor is the first width
at which T_max lags the ideal line by >= 3 steps.
Precisions: float16, float32, float64 (numpy), mpmath 25 and 40 digits. a in {1.3, 1.5, 2.0} (h = 0.239, 0.369, 0.631).

Prediction (written before the first run): delta_floor ~ C * eps^h, i.e. the slope of log(delta_floor) against log(eps)
equals h within 25% for each a. (Equivalently: usable decades ~ h * log10(1/eps).)
Bug fix after the first run (prediction unchanged): the probe line was centred exactly on the graph point, whose
lifetime is infinite, so the detector fired at the first level for every precision. The line is now offset.
"""
import json
import math

import mpmath as mp
import numpy as np

TH = 0.7
CAP = 600
EPS = {"float16": float(np.finfo(np.float16).eps), "float32": float(np.finfo(np.float32).eps),
       "float64": float(np.finfo(np.float64).eps), "mp25": 10.0 ** -25, "mp40": 10.0 ** -40}


def make(prec, a):
    if prec.startswith("mp"):
        mp.mp.dps = int(prec[2:])
        c, s = mp.cos(mp.mpf(TH)), mp.sin(mp.mpf(TH))
        A, two_pi = mp.mpf(a), 2 * mp.pi
        cosf, floor, conv = mp.cos, mp.floor, mp.mpf
    else:
        dt = getattr(np, prec)
        c, s = dt(math.cos(TH)), dt(math.sin(TH))
        A, two_pi = dt(a), dt(2 * math.pi)
        cosf, floor, conv = (lambda v: dt(np.cos(v))), (lambda v: dt(np.floor(v))), dt

    def lifetime(u1, u2):
        for t in range(CAP):
            x, y = c * u1 + s * u2, -s * u1 + c * u2       # p = R^T u
            if abs(y) > 10:
                return t
            x2 = 3 * x
            x2 = x2 - floor(x2)
            y2 = A * y + cosf(two_pi * x)
            u1, u2 = c * x2 - s * y2, s * x2 + c * y2     # u = R p
        return CAP
    return lifetime, conv, c, s


def graph_point(a, x0):
    mp.mp.dps = 150
    x = mp.mpf(x0)
    W = -sum(mp.cos(2 * mp.pi * ((mp.mpf(3) ** n * x) % 1)) / mp.mpf(a) ** (n + 1) for n in range(200))
    return x, W


def probe(prec, a, x0, wmin):
    gx, gy = graph_point(a, x0)
    lifetime, conv, c, s = make(prec, a)
    # u0 = R (gx, gy); direction = R e_y = (-s, c)
    u1 = conv(mp.cos(TH) * gx - mp.sin(TH) * gy) if prec.startswith("mp") else conv(float(mp.cos(TH) * gx - mp.sin(TH) * gy))
    u2 = conv(mp.sin(TH) * gx + mp.cos(TH) * gy) if prec.startswith("mp") else conv(float(mp.sin(TH) * gx + mp.cos(TH) * gy))
    off = 0.0137 * (1 + x0)        # bug fix: the centre must not sit exactly on the graph (it never settles)
    lo, hi = -0.25 + off, 0.25 + off
    W, TM = [], []
    while hi - lo > wmin:
        ss = [lo + (hi - lo) * i / 8 for i in range(9)]
        T = [lifetime(u1 + conv(si) * (-s), u2 + conv(si) * c) for si in ss]
        i = int(np.argmax(T))
        W.append(hi - lo)
        TM.append(T[i])
        lo, hi = ss[max(i - 1, 0)], ss[min(i + 1, 8)]
    return np.array(W), np.array(TM)


def floor_of(W, TM, lam_x):
    L = np.log(1 / W)
    ideal = TM[0] + (L - L[0]) / lam_x
    lag = ideal - TM
    j = np.where(lag >= 3)[0]
    return float(W[j[0]]) if len(j) else None


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    xs = rng.random(6)
    out = {}
    for a in (1.3, 1.5, 2.0):
        h = math.log(a) / math.log(3)
        out[str(a)] = {"h": h, "floors": {}}
        for prec, eps in EPS.items():
            wmin = max(eps ** h * 1e-4, 1e-60)
            fl = [floor_of(*probe(prec, a, x0, wmin), math.log(a)) for x0 in xs]
            fl = [f for f in fl if f is not None]
            out[str(a)]["floors"][prec] = {"eps": eps, "median_floor": float(np.median(fl)) if fl else None, "n": len(fl)}
            print(f"a {a} h {h:.3f} {prec:8s} eps {eps:.1e}: floor {np.median(fl) if fl else None} (n {len(fl)}), "
                  f"eps^h {eps ** h:.1e}", flush=True)
        E = [v["eps"] for v in out[str(a)]["floors"].values() if v["median_floor"]]
        F = [v["median_floor"] for v in out[str(a)]["floors"].values() if v["median_floor"]]
        slope = float(np.polyfit(np.log(E), np.log(F), 1)[0]) if len(E) >= 3 else None
        out[str(a)]["slope_logfloor_vs_logeps"] = slope
        print(f"==> a {a}: fitted slope {slope} vs h {h:.3f}", flush=True)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
