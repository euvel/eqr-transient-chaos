"""Regenerate every number quoted in README.md / note (except the survival-shape and slimness checks,
which have their own scripts). Run from the repository root: python analysis/key_numbers.py"""
import json

import numpy as np

from common import RES, f_obs, f_pred, kappa_valid, load_eps_runs, load_maps, spearman

maps = load_maps()

print("== 1. Lyapunov exponent and escape rate (settle-time maps, float32)")
lam = {r: p["lambda1"] for r, p in maps.items() if p["lambda1"] is not None}
v = np.array(list(lam.values()))
print(f"lambda1 over {len(v)} puzzles: median {np.median(v):.2f} /loop, IQR {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f},"
      f" range {v.min():.2f}-{v.max():.2f}")
kv = {r: p for r, p in maps.items() if kappa_valid(p) and p["lambda1"] is not None}
ks = np.array([p["kappa"] for p in kv.values()]); ls = np.array([p["lambda1"] for p in kv.values()])
print(f"kappa (valid fits, R^2 >= 0.97) over {len(kv)} puzzles: {ks.min():.3f}-{ks.max():.3f} /loop")
print(f"Spearman(lambda1, kappa) = {spearman(ls, ks):+.2f}   Spearman(kappa, human rating) = "
      f"{spearman(ks, np.array([p['rating'] for p in kv.values()])):+.2f}")
med = np.array([np.median(p["T"]) for p in kv.values()])
print(f"Spearman(lambda1, median settle) = {spearman(ls, med):+.2f}")

print("\n== 2. Registered results (from the kernels' own report.json)")
for run in ("eps_pairs_v5", "eps_pairs_v6"):
    r = json.load(open(RES / run / "report.json"))["result"]
    print(run, json.dumps({k: v for k, v in r.items() if k != "E2_slopes"}, default=float))

print("\n== 3. Exponent: measured slope of f(eps) vs kappa/lambda1 (v5 + v6, 9 puzzles)")
sl, kl = [], []
for run in ("eps_pairs_v5", "eps_pairs_v6"):
    for row, s in json.load(open(RES / run / "report.json"))["result"]["E2_slopes"].items():
        sl.append(s["slope"]); kl.append(s["kappa_over_lambda1"])
        print(f"  {run[-2:]} puzzle {row:>5}: slope {s['slope']:.3f}  kappa/lambda1 {s['kappa_over_lambda1']:.3f}  "
              f"ratio {s['slope'] / s['kappa_over_lambda1']:.2f}  ({s['n_levels']} levels)")
print(f"Spearman(slope, kappa/lambda1) over {len(sl)} puzzles = {spearman(np.array(sl), np.array(kl)):.2f}")

print("\n== 4. Amplitude: horizon law f = mean S(ln(0.1/d0)/lambda1), points with 0.05 <= f_pred <= 0.95")
jobs = load_eps_runs()
for tag, own in (("map S (registered)", False), ("each job's own 48 base settle times as S", True)):
    for run in ("eps_pairs_v5", "eps_pairs_v6", "both"):
        lr = []
        for j in jobs:
            if run != "both" and j["run"] != run:
                continue
            S = [float((np.array(j["Ta"]) > k).mean()) for k in range(151)] if own else None
            fp = f_pred(j, S)
            if 0.05 <= fp <= 0.95:
                lr.append(np.log(f_obs(j) / fp))
        lr = np.array(lr)
        print(f"  {tag:<42} {run:<13} n={len(lr):2d}  within 2x {np.mean(np.abs(lr) <= np.log(2)):4.0%}  "
              f"median |ln r| {np.median(np.abs(lr)):.2f}  bias {np.median(lr):+.2f}")
