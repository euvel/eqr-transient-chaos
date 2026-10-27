"""Shared loaders. Every analysis script reads only from results/ (paths relative to the repository root)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_PAIRS = 48


def load_maps():
    """Per puzzle (rows of Sudoku-Extreme test.csv): settle times, escape-rate fit, Lyapunov exponent.
    From the v3b and v4 runs (EqR, float32, 40x40 map of the demo 2-plane, cap 150)."""
    out = {}
    for run in ("maps_v3b", "maps_v4"):
        rep = json.load(open(RES / run / "report.json"))
        for row, m in rep["main"].items():
            ly = rep["lyap"].get(row, {})
            esc = m.get("escape") or {}
            T = np.load(RES / run / f"main_r{row}.npz")["T_dec"]
            out[int(row)] = {"run": run, "rating": m.get("rating"), "T": T,
                             "kappa": esc.get("kappa"), "kappa_r2": esc.get("r2"), "kappa_ci95": esc.get("kappa_ci95"),
                             "lambda1": ly.get("lambda1"), "lambda1_points": ly.get("lambda1_points", [])}
    return out


def kappa_valid(p):
    return p["kappa"] is not None and p["kappa_r2"] is not None and p["kappa_r2"] >= 0.97


def load_eps_runs():
    """eps-pair jobs from v5 and v6 (true float64), with each job's puzzle parameters attached."""
    jobs = []
    for run in ("eps_pairs_v5", "eps_pairs_v6"):
        rep = json.load(open(RES / run / "report.json"))
        P = json.load(open(RES / run / f"puzzles_{run[-2:]}.json"))
        for j in rep["jobs"].values():
            jobs.append(dict(j, run=run, puzzle=P[str(j["row"])]))
    return jobs


def S_of(S, t):
    """Linear interpolation of the survival curve S(k) = P(T > k) at real t."""
    if t <= 0:
        return 1.0
    lo = int(np.floor(t))
    if lo + 1 >= len(S):
        return S[-1]
    w = t - lo
    return (1 - w) * S[lo] + w * S[lo + 1]


def f_pred(job, S=None, thresh=0.1):
    """Horizon law: f = mean_i S(ln(thresh / d0_i) / lambda1)."""
    p = job["puzzle"]
    S = p["S"] if S is None else S
    t = np.log(thresh / np.asarray(job["d0"])) / p["lambda1"]
    return float(np.mean([S_of(S, x) for x in t]))


def f_obs(job, smoothed=True):
    k = job["k_differ"]
    return (k + 0.5) / (N_PAIRS + 1) if smoothed else k / N_PAIRS


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
