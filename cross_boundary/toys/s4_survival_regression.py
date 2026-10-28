"""S4: survival-regression Lyapunov spectroscopy on toys with FLUCTUATING stretching (pure numpy).

Idea: a trajectory survives only if it was not stretched much along the ESCAPE direction. Along natural (unkicked)
trajectories, compute local Gram-Schmidt (QR) exponents; over a window [A, A+W) of trajectories alive at A+W, regress
the remaining lifetime R = T - (A+W) jointly on the window stretches S_k = sum of local exponent k.
Prediction: the escape direction carries a negative coefficient beta ~ -1/lambda_esc; the others ~0.
  identified k* = most significant negative coefficient; lambda_x_hat = -1/beta_{k*}
  ROUGH iff k* > 1 (escape is not the fastest direction), FILAM iff k* = 1.

Toys (8-D: churn x1, x2 with nonuniform expansion; decision y; 5 stable s):
  x1' = 3 x1 + (1.2/2pi) sin(2pi x1) mod 1   (local stretch 1.8..4.2)
  x2' = 2 x2 + (0.8/2pi) sin(2pi x2) mod 1   (1.2..2.8)
  s'  = 0.5 s + 0.3 sin(2pi x1 + c) + 0.05 w y
  ROUGH-A: y' = a(x2) y + cos 2pi x1 + 0.5 cos 2pi x2, a(x2) = 1.5 exp(0.4 sin(2pi x2 + 0.7)); settled |y| > 10
  ROUGH-B: same with a(x1) = 1.5 exp(0.4 sin(2pi x1 + 0.7))   (confounded with lambda_1's fluctuations)
  FILAM:   y' = 0.5 y + forcing; settled when x1 in (0.4, 0.6) + 0.05 sin(2pi x2)
Expectations (written before the first run): ROUGH-A and ROUGH-B -> k* = 3 (the decision exponent, ~0.4 on average),
lambda_x_hat within 35% of the measured mean decision exponent, the S_1 coefficient not significant (|t| < 3);
FILAM -> k* = 1, lambda_x_hat within 35% of lambda_1.

Amendment 1 (2026-10-28, after the first run: rough_A k*=2 with a POSITIVE decision coefficient, rough_B k*=1,
filam no signal): the estimator regressed FUTURE lifetime on PAST stretching, but escape is nearly memoryless (window
survivors end up spread evenly near the saddle whatever their past). The thermodynamic principle weights SURVIVAL OF THE
WINDOW by exp(-S_esc over that window). New estimator: everyone alive at A gets window stretches over [A, A+W) (the toy
maps keep running after "settling"); logistic regression of [alive at A+W] on S_1..S_K jointly; k* = most negative
z-score. Expectations unchanged (lambda_x_hat is no longer estimated; identification only).
"""
import json

import numpy as np

rng = np.random.default_rng(0)
NS = 5
D = 3 + NS
C = rng.uniform(0, 2 * np.pi, NS)
Wy = rng.standard_normal(NS)
A, W, CAP, K = 6, 6, 200, 4


def a_of(x1, x2, kind):
    if kind == "rough_A":
        return 1.5 * np.exp(0.4 * np.sin(2 * np.pi * x2 + 0.7))
    if kind == "rough_B":
        return 1.5 * np.exp(0.4 * np.sin(2 * np.pi * x1 + 0.7))
    return np.full_like(x1, 0.5)


def da_of(x1, x2, kind):
    """(da/dx1, da/dx2)"""
    if kind == "rough_A":
        return 0 * x1, a_of(x1, x2, kind) * 0.4 * 2 * np.pi * np.cos(2 * np.pi * x2 + 0.7)
    if kind == "rough_B":
        return a_of(x1, x2, kind) * 0.4 * 2 * np.pi * np.cos(2 * np.pi * x1 + 0.7), 0 * x1
    return 0 * x1, 0 * x1


def step(P, kind):
    x1, x2, y, s = P[:, 0], P[:, 1], P[:, 2], P[:, 3:]
    out = np.empty_like(P)
    out[:, 0] = (3 * x1 + 1.2 / (2 * np.pi) * np.sin(2 * np.pi * x1)) % 1.0
    out[:, 1] = (2 * x2 + 0.8 / (2 * np.pi) * np.sin(2 * np.pi * x2)) % 1.0
    out[:, 2] = a_of(x1, x2, kind) * y + np.cos(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2)
    out[:, 3:] = 0.5 * s + 0.3 * np.sin(2 * np.pi * x1[:, None] + C) + 0.05 * Wy * y[:, None]
    return out


def settled(P, kind):
    if kind.startswith("rough"):
        return np.abs(P[:, 2]) > 10
    c = 0.5 + 0.05 * np.sin(2 * np.pi * P[:, 1])
    return np.abs(P[:, 0] - c) < 0.1


def jac(P, kind):
    n = len(P)
    x1, x2, y = P[:, 0], P[:, 1], P[:, 2]
    J = np.zeros((n, D, D))
    J[:, 0, 0] = 3 + 1.2 * np.cos(2 * np.pi * x1)
    J[:, 1, 1] = 2 + 0.8 * np.cos(2 * np.pi * x2)
    d1, d2 = da_of(x1, x2, kind)
    J[:, 2, 0] = d1 * y - 2 * np.pi * np.sin(2 * np.pi * x1)
    J[:, 2, 1] = d2 * y - np.pi * np.sin(2 * np.pi * x2)
    J[:, 2, 2] = a_of(x1, x2, kind)
    J[:, 3:, 0] = 0.6 * np.pi * np.cos(2 * np.pi * x1[:, None] + C)
    J[:, 3:, 2] = 0.05 * Wy
    J[:, 3:, 3:] = 0.5 * np.eye(NS)
    return J


def run(kind, n=300_000):
    P = np.column_stack([rng.random(n), rng.random(n), rng.uniform(-3, 3, n), rng.uniform(-1, 1, (n, NS))])
    T = np.full(n, CAP)
    alive = np.ones(n, bool)
    Q = np.linalg.qr(rng.standard_normal((n, D, K)))[0]
    loc = np.zeros((n, A + W, K))
    X = P.copy()
    for t in range(CAP):
        d = alive & settled(X, kind)
        T[d] = t
        alive &= ~d
        if t == A:
            atA = alive.copy()
        if t < A + W:  # QR on everything alive (t < A) or alive at A (window; Amendment 1)
            idx = np.where(alive if t < A else atA)[0]
            q, r = np.linalg.qr(jac(X[idx], kind) @ Q[idx])
            Q[idx] = q
            loc[idx, t] = np.log(np.abs(np.diagonal(r, axis1=1, axis2=2)))
        if not alive.any():
            break
        X = step(X, kind)
    Sw = loc[atA, A:A + W].sum(1)                          # window stretches S_k (everyone alive at A)
    Sw = (Sw - Sw.mean(0)) / Sw.std(0)                     # standardised: coefficients comparable across k
    yv = (T[atA] >= A + W).astype(float)                   # survived the window
    Xd = np.column_stack([np.ones(len(yv)), Sw])
    beta = np.zeros(Xd.shape[1])
    for _ in range(50):                                    # logistic regression (IRLS)
        p = 1 / (1 + np.exp(-Xd @ beta))
        H = Xd.T @ (Xd * (p * (1 - p))[:, None])
        beta += np.linalg.solve(H, Xd.T @ (yv - p))
    z = beta / np.sqrt(np.diag(np.linalg.inv(H)))
    k = int(np.argmin(z[1:]))
    return {"n_alive_at_A": int(atA.sum()), "survived_frac": round(float(yv.mean()), 3),
            "mean_local_exponents": loc[atA, A:A + W].mean((0, 1)).round(3).tolist(),
            "beta_std": beta[1:].round(3).tolist(), "z": z[1:].round(1).tolist(), "k_star": k + 1,
            "verdict": "ROUGH" if k > 0 else "FILAM"}


if __name__ == "__main__":
    out = {k: run(k) for k in ("rough_A", "rough_B", "filam")}
    for k, v in out.items():
        print(k, v)
    json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
