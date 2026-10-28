"""A1b: the contamination law. Same toy and probe as A1 at 40 digits (rounding floor negligible), but the probe
direction is R(e_y + c e_x)/norm: the exact cross direction contaminated by a fraction c of the churn direction.
Prediction (written before the run): delta_floor ~ C * c^(h/(1-h)); fitted slope of log floor vs log c within 25% of
h/(1-h) for a in {1.3, 1.5, 2.0} (h/(1-h) = 0.314, 0.585, 1.712).
"""
import json
import math

import mpmath as mp
import numpy as np

import a1_precision_law as A


def probe_c(a, x0, cc, wmin):
    gx, gy = A.graph_point(a, x0)
    lifetime, conv, c, s = A.make("mp40", a)
    u1, u2 = c * gx - s * gy, s * gx + c * gy
    dx, dy = mp.mpf(cc), mp.mpf(1)            # internal direction (c, 1), normalised, then rotated
    nrm = mp.sqrt(dx * dx + dy * dy)
    dx, dy = dx / nrm, dy / nrm
    v1, v2 = c * dx - s * dy, s * dx + c * dy
    off = mp.mpf(0.0137) * (1 + x0)
    lo, hi = mp.mpf(-0.25) + off, mp.mpf(0.25) + off
    W, TM = [], []
    while hi - lo > wmin:
        ss = [lo + (hi - lo) * i / 8 for i in range(9)]
        T = [lifetime(u1 + si * v1, u2 + si * v2) for si in ss]
        i = int(np.argmax(T))
        W.append(float(hi - lo))
        TM.append(T[i])
        lo, hi = ss[max(i - 1, 0)], ss[min(i + 1, 8)]
    return np.array(W), np.array(TM)


if __name__ == "__main__":
    xs = np.random.default_rng(1).random(6)
    out = {}
    for a in (1.3, 1.5, 2.0):
        h = math.log(a) / math.log(3)
        q = h / (1 - h)
        rows = {}
        for cc in (1e-4, 1e-8, 1e-12, 1e-16):
            wmin = max(cc ** q * 1e-4, 1e-30)
            fl = [A.floor_of(*probe_c(a, x0, cc, wmin), math.log(a)) for x0 in xs]
            fl = [f for f in fl if f is not None]
            rows[cc] = float(np.median(fl)) if fl else None
            print(f"a {a} c {cc:.0e}: floor {rows[cc]} (n {len(fl)}), c^(h/(1-h)) {cc ** q:.1e}", flush=True)
        C = [k for k, v in rows.items() if v]
        slope = float(np.polyfit(np.log(C), np.log([rows[k] for k in C]), 1)[0]) if len(C) >= 3 else None
        out[str(a)] = {"h": h, "h_over_1mh": q, "floors": {str(k): v for k, v in rows.items()}, "slope": slope}
        print(f"==> a {a}: fitted slope {slope} vs h/(1-h) {q:.3f}", flush=True)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
