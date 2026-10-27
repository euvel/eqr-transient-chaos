"""Build the v6 puzzle inputs (results/eps_pairs_v6/puzzles_v6.json) from the settle-time maps.

Recipe (identical to v5): S(k) = P(T_dec > k) from the 40x40 main map (k = 0..150), cap = max(16, ceil(1.5 x p99(T_dec))),
kappa = escape-rate fit and lambda1 = median Lyapunov exponent from the same run's report.json.
The recipe is checked against v5's stored entries for 1570 and 1982 before writing.
Run from the repository root: python experiments/03_eps_pairs_v6/make_puzzles_v6.py [--check]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "results/maps_v4"
rep = json.load(open(V4 / "report.json"))
cands = {c["row"]: c["question"] for c in
         json.load(open(ROOT / "experiments/01_settle_maps_and_lyapunov/candidates_v3.json"))["candidates"]}
v5 = json.load(open(ROOT / "results/eps_pairs_v5/puzzles_v5.json"))


def build(row):
    T = np.load(V4 / f"main_r{row}.npz")["T_dec"]
    return {"q": cands[row], "kappa": rep["main"][str(row)]["escape"]["kappa"],
            "lambda1": rep["lyap"][str(row)]["lambda1"],
            "S": [float((T > k).mean()) for k in range(151)],
            "cap": max(16, math.ceil(1.5 * np.percentile(T, 99)))}


for r in (1570, 1982):
    b, v = build(r), v5[str(r)]
    assert b["q"] == v["q"] and b["cap"] == v["cap"] and np.allclose(b["S"], v["S"]), r
    assert abs(b["kappa"] - v["kappa"]) < 1e-12 and abs(b["lambda1"] - v["lambda1"]) < 1e-12, r
out = {r: v5[r] for r in ("1570", "80", "1982")}
out["149"] = build(149)
target = ROOT / "results/eps_pairs_v6/puzzles_v6.json"
if "--check" in sys.argv:
    ok = json.load(open(target)) == json.loads(json.dumps(out))
    print("puzzles_v6.json reproduced exactly:", ok)
    sys.exit(0 if ok else 1)
json.dump(out, open(target, "w"))
