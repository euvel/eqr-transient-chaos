"""S5: escape-triggered averages (ETA): read the departure rate from the saddle off natural escaping trajectories.

No perturbations, no directions, no special precision. For natural trajectories with settle time T >= TMIN, average a
smooth observable aligned at the escape: m(tau) = <O(T - tau)>, tau = 1..TAU. Far from the escape m -> baseline (the
saddle average). Near it, |m(tau) - baseline| ~ A exp(-lambda_dep tau), with lambda_dep the rate of departure along the
escape direction: lambda_x for a rough boundary, lambda_1 for a filamentary one.
Fit: log|m - baseline| against tau over tau in [3, last tau with signal > 3 SE] (tau <= 2 excluded: the escape itself);
baseline = mean of m over the last 5 taus. If the signal at tau = 3 is < 3 SE, the departure is ABRUPT (rate > ~1).

Toys (churn x1' = 3 x1 + 0.1 sin 2pi x2 mod 1 or logistic, x2' = 2 x2 mod 1, decision y):
  ROUGH(a):  y' = a y + cos 2pi x1 + 0.5 cos 2pi x2, settled |y| > 10; a = 1.2, 1.5 (kappa = lambda_x = ln a = 0.18, 0.41)
  FILAM(mu): x1' = mu x1 (1 - x1) + 0.02 sin 2pi x2 (smooth escape along the unstable direction), y' = 0.5 y + forcing,
             settled |x1| > 5; mu = 4.2, 4.6; lambda_1 = the logistic repeller's exponent (measured)
Observables (same form for all toys): O_esc = tanh(|e|/3) with e the escaping coordinate (y or x1 - 0.5), and a
generic mix O_gen = O_esc + 0.5 sin(2 pi x2 + 1) + 0.3 cos(2 pi y / 7).
Prediction (written before the run): ROUGH -> lambda_dep within 30% of kappa for both a (it tracks kappa);
FILAM -> ABRUPT or lambda_dep >= 0.8 lambda_1. Both observables.
"""
import json

import numpy as np

rng = np.random.default_rng(6)
TAU, TMIN, CAP = 18, 30, 400


def simulate(kind, par, n):
    x1, x2 = rng.random(n), rng.random(n)
    y = rng.uniform(-2, 2, n)
    T = np.full(n, CAP)
    alive = np.ones(n, bool)
    hist = np.zeros((CAP, n, 3), np.float32)
    lam_sum, lam_n = 0.0, 0
    for t in range(CAP):
        hist[t, :, 0], hist[t, :, 1], hist[t, :, 2] = x1, x2, y
        if kind == "rough":
            d = alive & (np.abs(y) > 10)
        else:
            d = alive & (np.abs(x1) > 5)
        T[d] = t
        alive &= ~d
        if not alive.any():
            break
        if kind == "rough":
            x1, x2, y = ((3 * x1 + 0.1 * np.sin(2 * np.pi * x2)) % 1.0, (2 * x2) % 1.0,
                         par * y + np.cos(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2))
        else:
            ok = alive & (t > 5)
            lam_sum += np.log(np.abs(par * (1 - 2 * x1[ok]))).sum()
            lam_n += ok.sum()
            x1, x2, y = (par * x1 * (1 - x1) + 0.02 * np.sin(2 * np.pi * x2), (2 * x2) % 1.0,
                         0.5 * y + np.cos(2 * np.pi * np.clip(x1, -5, 5)) + 0.5 * np.cos(2 * np.pi * x2))
            x1 = np.clip(x1, -1e3, 1e3)
    lam1 = lam_sum / max(lam_n, 1) if kind == "filam" else np.log(3)
    return T, hist, lam1


def observables(h, kind):
    x1, x2, y = h[..., 0], h[..., 1], h[..., 2]
    e = y if kind == "rough" else x1 - 0.5
    o_esc = np.tanh(np.abs(e) / 3)
    o_gen = o_esc + 0.5 * np.sin(2 * np.pi * x2 + 1) + 0.3 * np.cos(2 * np.pi * np.clip(y, -50, 50) / 7)
    return {"esc": o_esc, "gen": o_gen}


def kappa_of(T):
    S = np.array([(T > j).mean() for j in range(CAP)])
    m = (S <= 0.5) & (S >= 1e-4)
    return float(-np.polyfit(np.arange(CAP)[m], np.log(S[m]), 1)[0])


def eta(T, hist, kind):
    idx = np.where((T >= TMIN) & (T < CAP))[0]
    taus = np.arange(1, TAU + 1)
    out = {"n_events": int(len(idx))}
    for name in ("esc", "gen"):
        vals = np.array([[observables(hist[T[i] - tau, i], kind)[name] for tau in taus] for i in idx])
        m, se = vals.mean(0), vals.std(0) / np.sqrt(len(idx))
        base = m[-5:].mean()
        sig = np.abs(m - base)
        ok = sig > 3 * se
        if not ok[2]:
            out[name] = {"m": m.round(4).tolist(), "verdict": "ABRUPT", "lambda_dep": None}
            continue
        last = 2
        while last + 1 < len(taus) - 5 and ok[last + 1]:
            last += 1
        sel = slice(2, last + 1)
        lam = float(-np.polyfit(taus[sel], np.log(sig[sel]), 1)[0]) if last >= 4 else None
        out[name] = {"m": m.round(4).tolist(), "fit_taus": [3, int(taus[last])], "lambda_dep": lam,
                     "verdict": "GRADUAL" if lam is not None else "ABRUPT"}
    return out


if __name__ == "__main__":
    res = {}
    for kind, pars in (("rough", (1.2, 1.5)), ("filam", (4.2, 4.6))):
        for p in pars:
            T, hist, lam1 = simulate(kind, p, 400_000)
            r = {"kappa": round(kappa_of(T), 3), "lambda1": round(float(lam1), 3), **eta(T, hist, kind)}
            res[f"{kind}_{p}"] = r
            print(f"{kind} {p}: kappa {r['kappa']}, lambda1 {r['lambda1']}, events {r['n_events']}; "
                  f"esc: {r['esc']['verdict']} {r['esc']['lambda_dep']}; gen: {r['gen']['verdict']} {r['gen']['lambda_dep']}",
                  flush=True)
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
