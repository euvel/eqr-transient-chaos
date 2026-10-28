"""S0: calibrate cross-boundary estimators on two toy maps with KNOWN answers (pure numpy, no model).

Both toys are single-outcome transients like EqR (every point eventually "settles"; T = steps until settled), and they
are tuned to have the SAME lambda_max = ln 3, the SAME escape rate kappa = ln 1.5 and hence the SAME uncertainty exponent
alpha = kappa/lambda_max = 0.369. They differ only in the cross-boundary exponent lambda_x:

  ROUGH  (Weierstrass type):  x' = 3x mod 1,  y' = 1.5 y + cos(2 pi x);  settled when |y| > 10.
         Repeller = the graph y = W(x) (nowhere differentiable). lambda_x = ln 1.5 = 0.405 < lambda_max.
  FILAM  (leaky expanding map): x' = 3x mod 1,  y' = 0.5 y + cos(2 pi x); settled when x lands in (1/3, 2/3).
         Repeller = Cantor set in x times all y (smooth vertical leaves). lambda_x = lambda_max = ln 3 = 1.099.

Expectations, written before the first run:
  E1 kappa (exponential fit to survival of uniform starts) ~ 0.405 in both toys (within 10%).
  E2 Line-PIM: the lifetime slope dT/dln(1/delta) along RANDOM lines is ~ 1/ln3 = 0.91 in BOTH toys (within 15%).
     If true, the bisection-lifetime idea from the brainstorm cannot tell rough from filamentary (as Bodai & Lucarini
     say for any measurement along a line) and is dropped as a discriminator.
  E3 Lyapunov spectrum on long-lived (saddle) trajectories: ROUGH (1.099, 0.405); FILAM (1.099, -0.69).
     Rough requires a second POSITIVE exponent; its absence rules rough out.
  E4 Direction-resolved lifetime at saddle points: perturbing along the y direction (the cross direction in ROUGH,
     the stable direction in FILAM) gives slope 1/0.405 = 2.47 in ROUGH and no change in T in FILAM; along x, 0.91
     in both. Rough <=> some unstable direction v_k (k >= 2) gives slope 1/lambda_k.

Amendment 1 (2026-10-27, after the first run; E1-E3 matched, E4 did not: ROUGH y-slope 0.90):
  E4 perturbed the PIM end point z directly. z is ~1e-12 from the boundary ALONG ITS LINE, but on a Hoelder graph that
  is ~(1e-12)^0.37 ~ 1e-5 away vertically, so small y-probes never straddle the boundary. E4 now runs PIM along a line
  through z in each probe direction (direction-resolved PIM). Expectations unchanged.
Amendment 2 (same day, after the second run; y-slope 2.35 matched, x/random slopes ~0):
  z was refined to T ~ 36, which is already the float64 floor for x-expansion (3^33 ~ 1e16), so probes with any x
  component could not climb further. z is now taken from a coarser PIM (width 1e-3). Expectations unchanged.
"""
import json
import numpy as np

LN3, LNA = np.log(3.0), np.log(1.5)
CAP = 200
rng = np.random.default_rng(0)


def settle_time(x, y, toy):
    """Steps until settled (vectorised). CAP if never."""
    x, y = np.array(x, float), np.array(y, float)
    T = np.full(x.shape, CAP)
    alive = np.ones(x.shape, bool)
    for t in range(CAP):
        done = alive & ((np.abs(y) > 10) if toy == "rough" else ((x > 1 / 3) & (x < 2 / 3)))
        T[done] = t
        alive &= ~done
        if not alive.any():
            break
        x, y = (3 * x) % 1.0, (1.5 if toy == "rough" else 0.5) * y + np.cos(2 * np.pi * x)
    return T


def sample_box(n):
    return rng.random(n), rng.uniform(-4, 4, n)


def kappa(toy):
    T = settle_time(*sample_box(200_000), toy)
    k = np.arange(CAP)
    S = np.array([(T > j).mean() for j in k])
    m = (S <= 0.9) & (S >= 1e-3)
    g = np.polyfit(k[m], np.log(S[m]), 1)[0]
    return -g


def pim_line(p0, p1, toy, stop=1e-12, n=9):
    """PIM: keep the sub-segment around the longest-lived sample. Returns (widths, Tmax, final point)."""
    a, b = 0.0, 1.0
    d = p1 - p0
    W, TM = [], []
    while (b - a) * np.linalg.norm(d) > stop:
        s = np.linspace(a, b, n)
        P = p0[:, None] + d[:, None] * s
        T = settle_time(P[0], P[1], toy)
        i = int(np.argmax(T))
        W.append((b - a) * np.linalg.norm(d))
        TM.append(int(T[i]))
        a, b = s[max(i - 1, 0)], s[min(i + 1, n - 1)]
    s = 0.5 * (a + b)
    return np.array(W), np.array(TM), p0 + d * s


def slope(widths, T, lo=1e-3, hi=1e-11):
    m = (widths <= lo) & (widths >= hi)
    return np.polyfit(np.log(1 / widths[m]), T[m], 1)[0]


def jac(x, y, toy):
    b = 1.5 if toy == "rough" else 0.5
    return np.array([[3.0, 0.0], [-2 * np.pi * np.sin(2 * np.pi * x), b]])


def spectrum(z, toy, steps):
    """Lyapunov exponents along the first `steps` iterates of z (QR)."""
    x, y = z
    Q = np.eye(2)
    s = np.zeros(2)
    for _ in range(steps):
        Q, R = np.linalg.qr(jac(x, y, toy) @ Q)
        s += np.log(np.abs(np.diag(R)))
        x, y = (3 * x) % 1.0, (1.5 if toy == "rough" else 0.5) * y + np.cos(2 * np.pi * x)
    return s / steps


def direction_slope(z, v, toy, half=0.05):
    """Direction-resolved PIM: bisect along the line z + s v, |s| <= half, and fit the lifetime slope."""
    W, TM, _ = pim_line(z - half * v, z + half * v, toy)
    if np.ptp(TM) == 0:
        return float("nan")  # T does not depend on this direction: no boundary crossed
    return slope(W, TM)


def run(toy, nlines=40):
    out = {"kappa": kappa(toy)}
    line_slopes, specs, sx, sy, srand = [], [], [], [], []
    for _ in range(nlines):
        p0 = np.array([rng.random(), rng.uniform(-4, 4)])
        ang = rng.uniform(0, np.pi)
        p1 = p0 + 1.0 * np.array([np.cos(ang), np.sin(ang)])
        W, TM, z = pim_line(p0, p1, toy)
        if TM.max() < 15:
            continue  # line missed the repeller
        line_slopes.append(slope(W, TM))
        z = pim_line(p0, p1, toy, stop=1e-3)[2]  # Amendment 2: leave room below the float64 floor
        specs.append(spectrum(z, toy, max(5, int(TM.max()) - 3)))
        sx.append(direction_slope(z, np.array([1.0, 0.0]), toy))
        sy.append(direction_slope(z, np.array([0.0, 1.0]), toy))
        a = rng.uniform(0, np.pi)
        srand.append(direction_slope(z, np.array([np.cos(a), np.sin(a)]), toy))
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
    out.update(n_lines=len(line_slopes), line_slope_median=med(line_slopes),
               line_slope_iqr=[float(np.nanpercentile(line_slopes, q)) for q in (25, 75)],
               spectrum_median=np.median(np.array(specs), axis=0).tolist(),
               dir_slope_x=med(sx), dir_slope_y=med(sy), dir_slope_y_nan_frac=float(np.mean(np.isnan(sy))),
               dir_slope_random=med(srand))
    return out


if __name__ == "__main__":
    res = {toy: run(toy) for toy in ("rough", "filam")}
    res["reference"] = {"1/ln3": 1 / LN3, "1/ln1.5": 1 / LNA, "ln1.5": LNA, "ln3": LN3}
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
