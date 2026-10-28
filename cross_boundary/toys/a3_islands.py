"""A3: topology test ("islands vs stripes") in 2-D slices of settle-time maps (toys, numpy, float64).

Toys (3-D: churn x1, x2 on the torus, decision y):
  churn  x1' = 3 x1 + 0.1 sin(2 pi x2) mod 1,  x2' = 2 x2 mod 1
  ROUGH  y' = 1.5 y + cos 2pi x1 + 0.5 cos 2pi x2; settled |y| > 10   -> boundary = Hoelder graph y = W(x1, x2)
  FILAM  y' = 0.5 y + same forcing; settled when x1 in (1/3, 2/3) + 0.05 sin 2pi x2  -> smooth sheets x1 = phi_k(x2)
Slice: a random 2-D plane through a boundary point (found by bisection along a random line). Zoom windows of side
L in {0.5, 0.05, 0.005, 0.0005}, 256 x 256 settle-time maps. Binary field B = T >= t*, t* = 80th percentile of T in the
window. "Islands" = connected components of B or of its complement that do not touch the window border (8-connectivity).
Score per window: number of islands with >= 4 pixels, and the fraction of B's area inside islands.

Prediction (written before the first run), median over 6 slices:
  ROUGH: >= 3 islands at every depth and island-area fraction >= 10% at every depth;
  FILAM: <= 1 island and island-area fraction < 2% at depths L <= 0.05 (smooth sheets look like stripes when zoomed).
"""
import json

import numpy as np
from scipy import ndimage

rng = np.random.default_rng(4)
CAP, N = 300, 256


def settle(P, kind):
    x1, x2, y = P[..., 0].copy(), P[..., 1].copy(), P[..., 2].copy()
    T = np.full(x1.shape, CAP)
    alive = np.ones(x1.shape, bool)
    for t in range(CAP):
        if kind == "rough":
            d = alive & (np.abs(y) > 10)
        else:
            c = 0.05 * np.sin(2 * np.pi * x2)
            d = alive & (x1 > 1 / 3 + c) & (x1 < 2 / 3 + c)
        T[d] = t
        alive &= ~d
        if not alive.any():
            break
        a = 1.5 if kind == "rough" else 0.5
        x1, x2, y = ((3 * x1 + 0.1 * np.sin(2 * np.pi * x2)) % 1.0, (2 * x2) % 1.0,
                     a * y + np.cos(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2))
    return T


def boundary_point(kind):
    p0 = np.array([rng.random(), rng.random(), rng.uniform(-2, 2)])
    d = rng.standard_normal(3)
    d /= np.linalg.norm(d)
    lo, hi = -0.5, 0.5
    for _ in range(45):  # PIM along the line to ~1e-13
        s = np.linspace(lo, hi, 9)
        T = settle(p0[None] + s[:, None] * d[None], kind)
        i = int(np.argmax(T))
        lo, hi = s[max(i - 1, 0)], s[min(i + 1, 8)]
    return p0 + 0.5 * (lo + hi) * d


def window(kind, z, e1, e2, L):
    g = (np.arange(N) / (N - 1) - 0.5) * L
    S1, S2 = np.meshgrid(g, g, indexing="ij")
    P = z[None, None] + S1[..., None] * e1 + S2[..., None] * e2
    return settle(P, kind)


def islands(T):
    tstar = np.percentile(T, 80)
    B = T >= tstar
    out_n, out_area = 0, 0
    st = np.ones((3, 3), int)
    for field in (B, ~B):
        lab, n = ndimage.label(field, structure=st)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for k, sz in enumerate(sizes, start=1):
            if k not in border and sz >= 4:
                out_n += 1
                if field is B:
                    out_area += sz
    return out_n, out_area / max(B.sum(), 1)


if __name__ == "__main__":
    res = {}
    for kind in ("rough", "filam"):
        rows = {L: [] for L in (0.5, 0.05, 0.005, 0.0005)}
        for _ in range(6):
            z = boundary_point(kind)
            Q = np.linalg.qr(rng.standard_normal((3, 2)))[0]
            for L in rows:
                rows[L].append(islands(window(kind, z, Q[:, 0], Q[:, 1], L)))
        res[kind] = {str(L): {"islands_median": float(np.median([r[0] for r in v])),
                              "area_frac_median": round(float(np.median([r[1] for r in v])), 3),
                              "per_slice": v} for L, v in rows.items()}
        for L, v in res[kind].items():
            print(kind, "L", L, "islands", v["islands_median"], "island-area fraction", v["area_frac_median"],
                  "per slice", [r[0] for r in v["per_slice"]], flush=True)
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
