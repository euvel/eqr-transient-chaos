"""S0b: BLIND calibration of the cross-boundary pipeline on 30-D toys with a hidden basis (pure numpy).

The pipeline sees only the map F(u) and its Jacobian in a randomly rotated basis u = R p, plus a black-box settle time.
It does not know which coordinates are churn, decision or stable.

Internal coordinates p = (x1, x2, y, s_1..s_27):
  churn     x1' = 3 x1 + 0.1 sin(2 pi x2)  mod 1,   x2' = 2 x2  mod 1          (exponents ~ln3, ln2, two positive)
  stable    s_j' = 0.5 s_j + 0.3 sin(2 pi x1 + c_j) + 0.05 w_j y               (27 x ln0.5)
  ROUGH     y' = 1.5 y + cos(2 pi x1) + 0.5 cos(2 pi x2);   settled when |y| > 10.
            Decision instability ln1.5 = 0.405 < churn: rough. Truth lambda_x = 0.405, kappa = 0.405.
  FILAM     y' = 0.5 y + (same forcing);   settled when x1 in (1/3, 2/3) + 0.05 sin(2 pi x2).
            Escape only through the fastest direction; the outcome ALSO depends smoothly on the second chaotic
            direction x2 (adversarial). Truth lambda_x = lambda_max = 1.099.

Pipeline (identical for both toys, blind):
  1. 100k random starts; keep the 12 longest-lived trajectories.
  2. Lyapunov spectrum (K = 5) by QR along each trajectory's middle part; CLVs by Ginelli's backward iteration.
  3. At the middle point of each trajectory, direction-resolved PIM along each unstable CLV (lambda_k > 0.05) and along
     a random direction; fit the lifetime slope s_k = dT/dln(1/delta).
  4. lambda_x_hat = 1 / max_k median(s_k) over unstable CLVs.

Why max: on a rough boundary every unstable direction with lambda_k >= lambda_x gives s_k = 1/lambda_k (Hoelder exponent
lambda_x/lambda_k), and weaker ones give 1/lambda_x; on a filamentary boundary every unstable direction gives 1/lambda_1.
So the largest slope is 1/lambda_x in both cases.

Expectations (written before the first run):
  B1 spectrum: ROUGH (1.10, 0.69, 0.41, -0.69, -0.69); FILAM (1.10, 0.69, -0.69, -0.69, -0.69), within 10%.
  B2 random-direction slope ~ 1/lambda_1 = 0.91 in both (within 20%).
  B3 ROUGH slopes along CLV1, CLV2, CLV3 ~ 0.91, 1.44, 2.47; lambda_x_hat = 0.405 within 15%.
  B4 FILAM slopes along CLV1, CLV2 ~ 0.91, 0.91; lambda_x_hat = 1.10 within 15%.
  B5 Decision rule: ROUGH iff lambda_x_hat / lambda_1 < 0.7. Must say ROUGH for ROUGH and FILAM for FILAM.

Amendment 1 (2026-10-27, after the first run: B1 passed; ROUGH CLV3 slope missing in 92% of probes and the verdict was
FILAM for both toys):
  Diagnosis: float64 rounding injects ~1e-16 noise in ALL directions, which grows at lambda_1. A probe at width delta along
  the cross direction needs ln(1/delta)/lambda_x steps; it stays readable only while
  ln(1/delta) < ln(1e16) * lambda_x/lambda_1, i.e. ~16*alpha usable decades. (S0 escaped this because its probe line did
  not touch the churn coordinate.) The first run fitted down to 1e-11, far past that bound. Changes, all blind:
  (a) fit window: widths in [exp(-0.8 * 36.8 * alpha_hat), 1e-1] with alpha_hat = kappa_hat/lambda_1_hat (measured),
      requiring >= 1.5 decades;
  (b) probe point late on the trajectory (8 steps before settling), so the remaining lifetime adds little noise;
      CLVs computed at the middle and carried forward with the Jacobian (they are covariant);
  (c) half-width adaptive: 0.05, x4 until the coarse line shows T variation, max 3.
  Expectations B1-B5 unchanged.
"""
import json
import numpy as np

N = 30
CAP = 120
rng = np.random.default_rng(0)
R, _ = np.linalg.qr(rng.standard_normal((N, N)))  # hidden basis
C = rng.uniform(0, 2 * np.pi, 27)
Wy = rng.standard_normal(27)


def step(P, toy):
    x1, x2, y, s = P[:, 0], P[:, 1], P[:, 2], P[:, 3:]
    a = 1.5 if toy == "rough" else 0.5
    out = np.empty_like(P)
    out[:, 0] = (3 * x1 + 0.1 * np.sin(2 * np.pi * x2)) % 1.0
    out[:, 1] = (2 * x2) % 1.0
    out[:, 2] = a * y + np.cos(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2)
    out[:, 3:] = 0.5 * s + 0.3 * np.sin(2 * np.pi * x1[:, None] + C) + 0.05 * Wy * y[:, None]
    return out


def settled(P, toy):
    if toy == "rough":
        return np.abs(P[:, 2]) > 10
    c = 0.05 * np.sin(2 * np.pi * P[:, 1])
    return (P[:, 0] > 1 / 3 + c) & (P[:, 0] < 2 / 3 + c)


def jac(p, toy):
    J = np.zeros((N, N))
    x1, x2, y = p[0], p[1], p[2]
    J[0, 0], J[0, 1] = 3.0, 0.2 * np.pi * np.cos(2 * np.pi * x2)
    J[1, 1] = 2.0
    J[2, 0], J[2, 1], J[2, 2] = -2 * np.pi * np.sin(2 * np.pi * x1), -np.pi * np.sin(2 * np.pi * x2), (
        1.5 if toy == "rough" else 0.5)
    J[3:, 0] = 0.6 * np.pi * np.cos(2 * np.pi * x1 + C)
    J[3:, 2] = 0.05 * Wy
    J[3:, 3:] = 0.5 * np.eye(27)
    return J


# ---- black box seen by the pipeline (rotated basis) ----
def to_int(U):
    return U @ R  # rows: p = R^T u


def to_ext(P):
    return P @ R.T


def settle_time(U, toy):
    P = to_int(np.atleast_2d(U))
    T = np.full(len(P), CAP)
    alive = np.ones(len(P), bool)
    for t in range(CAP):
        d = alive & settled(P, toy)
        T[d] = t
        alive &= ~d
        if not alive.any():
            break
        P = step(P, toy)
    return T


def F(u, toy):
    return to_ext(step(to_int(u[None]), toy))[0]


def J_ext(u, toy):
    return R @ jac(to_int(u[None])[0], toy) @ R.T


# ---- pipeline ----
def random_starts(n, toy):
    P = np.column_stack([rng.random(n), rng.random(n), rng.uniform(-3, 3, n), rng.uniform(-1, 1, (n, 27))])
    return to_ext(P)


def pim_line(p0, p1, toy, stop=1e-12, n=9):
    a, b = 0.0, 1.0
    d = p1 - p0
    L = np.linalg.norm(d)
    W, TM = [], []
    while (b - a) * L > stop:
        s = np.linspace(a, b, n)
        T = settle_time(p0[None] + s[:, None] * d[None], toy)
        i = int(np.argmax(T))
        W.append((b - a) * L)
        TM.append(int(T[i]))
        a, b = s[max(i - 1, 0)], s[min(i + 1, n - 1)]
    return np.array(W), np.array(TM)


def lifetime_slope(z, v, toy, wmin, half=0.05):
    """Direction-resolved PIM along z + s v; adaptive half-width; fit over widths in [wmin, 1e-1] (Amendment 1)."""
    while True:
        W, TM = pim_line(z - half * v, z + half * v, toy, stop=wmin / 10)
        if np.ptp(TM) > 0 or half >= 3:
            break
        half *= 4
    m = (W <= 1e-1) & (W >= wmin)
    if m.sum() < 4 or np.ptp(TM[m]) == 0:
        return float("nan")
    return float(np.polyfit(np.log(1 / W[m]), TM[m], 1)[0])


def spectrum_and_clvs(traj, toy, K=5, skip=4):
    """QR forward over the whole trajectory; Ginelli backward; returns middle-part exponents and CLVs at index mid."""
    n = len(traj) - 1
    Q = np.linalg.qr(rng.standard_normal((N, K)))[0]
    Qs, Rs, loc = [Q], [], []
    for t in range(n):
        Q, Rm = np.linalg.qr(J_ext(traj[t], toy) @ Q)
        sg = np.sign(np.diag(Rm))
        Q, Rm = Q * sg, Rm * sg[:, None]
        Qs.append(Q)
        Rs.append(Rm)
        loc.append(np.log(np.abs(np.diag(Rm))))
    loc = np.array(loc)
    lam = loc[skip:n - skip].mean(0)
    mid = n // 2
    Cm = np.eye(K)
    for t in range(n - 1, mid - 1, -1):  # Q_{t+1} R_t = J_t Q_t  =>  C_t = R_t^{-1} C_{t+1}
        Cm = np.linalg.solve(Rs[t], Cm)
        Cm /= np.linalg.norm(Cm, axis=0)
    V = Qs[mid] @ Cm
    probe = max(mid, n - 8)
    for t in range(mid, probe):  # covariant: carry forward with the Jacobian (Amendment 1b)
        V = J_ext(traj[t], toy) @ V
        V /= np.linalg.norm(V, axis=0)
    return lam, V, probe


def run(toy, nstart=100_000, ntraj=12, K=5):
    U = random_starts(nstart, toy)
    T = settle_time(U, toy)
    idx = np.argsort(T)[::-1][:ntraj]
    kappa_S = np.array([(T > j).mean() for j in range(CAP)])
    m = (kappa_S <= 0.9) & (kappa_S >= 1e-3)
    kappa = float(-np.polyfit(np.arange(CAP)[m], np.log(kappa_S[m]), 1)[0])
    lams, slopes, rnd, wmins = [], [], [], []
    for i in idx:
        traj = [U[i]]
        for _ in range(T[i]):
            traj.append(F(traj[-1], toy))
        lam, V, mid = spectrum_and_clvs(traj, toy, K)
        lams.append(lam)
        wmin = max(np.exp(-0.8 * 36.8 * kappa / lam[0]), 1e-11)  # Amendment 1a (blind: measured kappa, lambda_1)
        wmins.append(wmin)
        slopes.append([lifetime_slope(traj[mid], V[:, k], toy, wmin) if lam[k] > 0.05 else float("nan")
                       for k in range(K)])
        v = rng.standard_normal(N)
        rnd.append(lifetime_slope(traj[mid], v / np.linalg.norm(v), toy, wmin))
    lams, slopes = np.array(lams), np.array(slopes)
    lam_med = np.median(lams, 0)
    s_med = np.array([np.nanmedian(slopes[:, k]) if np.isfinite(slopes[:, k]).any() else np.nan for k in range(K)])
    smax = np.nanmax(s_med[lam_med > 0.05])
    lx = 1 / smax
    return {"kappa": kappa, "T_longest": T[idx].tolist(), "spectrum": lam_med.round(3).tolist(),
            "clv_slopes": [None if np.isnan(x) else round(float(x), 3) for x in s_med],
            "clv_slope_nan_frac": np.isnan(slopes).mean(0).round(2).tolist(),
            "random_slope": float(np.nanmedian(rnd)), "fit_wmin": float(np.median(wmins)), "lambda_x_hat": float(lx),
            "ratio_to_lambda1": float(lx / lam_med[0]), "verdict": "ROUGH" if lx / lam_med[0] < 0.7 else "FILAM"}


if __name__ == "__main__":
    res = {toy: run(toy) for toy in ("rough", "filam")}
    print(json.dumps(res, indent=1))
    json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
