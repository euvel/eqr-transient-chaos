"""Apply PREREG_cross_boundary_eqr.md (Stage 1 and Stage 2) to S2c outputs.

Usage: s2_analyze.py out/s2c_r206.npz out/s2c_r814.npz ...
"""
import json
import sys
from pathlib import Path

import numpy as np

KAPPA = {206: 0.054, 814: 0.118, 225: 0.142, 80: 0.246, 1570: 0.122, 1216: 0.049}  # saved maps (report.json)
rng = np.random.default_rng(0)


def estimate(local, start=6, block=8, nboot=2000):
    L = local[start:]
    L = L[~np.isnan(L).any(1)]
    nb = len(L) // block
    lam = L.mean(0)
    blocks = np.array([L[i * block:(i + 1) * block].mean(0) for i in range(nb)])
    boot = np.array([blocks[rng.integers(0, nb, nb)].mean(0) for _ in range(nboot)])
    return lam, boot.std(0), len(L)


def main(paths):
    P = {}
    for p in paths:
        d = np.load(p)
        row = int(d["row"])
        lam, se, n = estimate(d["local"])
        P[row] = {"lam": lam, "se": se, "n": n, "kappa": KAPPA[row], "mismatch": float(d["max_mismatch"])}
    out = {"puzzles": {r: {"n_window": v["n"], "kappa": v["kappa"], "lambda": v["lam"].round(3).tolist(),
                           "se": v["se"].round(3).tolist(), "max_mismatch": v["mismatch"]} for r, v in P.items()}}
    usable = {r: v for r, v in P.items() if v["n"] >= 24}
    out["usable"] = sorted(usable)
    # Stage 1
    pos2 = {r: bool(v["lam"][1] > 0.03 and v["lam"][1] > 2 * v["se"][1]) for r, v in usable.items()}
    out["stage1_lambda2_positive"] = pos2
    if usable and not all(pos2.values()):
        out["verdict"] = "FILAMENTARY (Stage 1)"
        return out
    # Stage 2
    res = {r: v for r, v in usable.items() if v["lam"][-1] < v["kappa"] - 0.03}
    out["stage2_resolved"] = sorted(res)
    out["stage2_unresolved_K_too_small"] = sorted(set(usable) - set(res))
    if len(res) < 3:
        out["verdict"] = "PENDING (fewer than 3 resolved puzzles)"
        return out
    pos, se_pos, ok, removed = {}, {}, {}, {}
    for r, v in res.items():
        m = (v["lam"] > 0.03) & (v["lam"] > 2 * v["se"])
        pos[r], se_pos[r] = v["lam"][m], v["se"][m]
        k = int(np.argmin(np.abs(pos[r] - v["kappa"])))
        ok[r] = bool(abs(pos[r][k] - v["kappa"]) <= max(0.1 * v["kappa"] + 0.03, 2 * se_pos[r][k]))
        removed[r] = (np.delete(pos[r], k), np.delete(se_pos[r], k))

    def agree(A):
        rows = list(A)
        worst = 0.0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (a, sa), (b, sb) = A[rows[i]], A[rows[j]]
                if len(a) != len(b):
                    return False, float("inf")
                tol = np.maximum(0.08, 3 * np.sqrt(sa ** 2 + sb ** 2))
                worst = max(worst, float(np.max(np.abs(a - b) - tol)) if len(a) else -1.0)
        return worst <= 0, worst

    a_out, w_out = agree(removed)
    a_all, w_all = agree({r: (pos[r], se_pos[r]) for r in res})
    out.update(stage2_match_ok=ok, n_positive={r: int(len(p)) for r, p in pos.items()},
               agree_after_removal=a_out, excess_after_removal=w_out, agree_full=a_all, excess_full=w_all)
    out["verdict"] = ("ROUGH-SKEW" if all(ok.values()) and a_out else "FILAM-CONSISTENT" if a_all else "NOT-SKEW")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1:])
    print(json.dumps(o, indent=1, default=str))
    json.dump(o, open(Path(sys.argv[1]).parent / "s2_verdict.json", "w"), indent=1, default=str)
