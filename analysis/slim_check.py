"""Does the eps-pair power law steepen at small eps ("slim fractal": alpha -> 1)?
Criteria fixed before running: per puzzle (v5 + v6), usable levels = 0.03 < f_obs < 0.97 with >= 3 levels.
Local slopes between consecutive usable levels; curvature c = slope(smallest-eps pair) - slope(largest-eps pair).
Bootstrap over pairs within each job (2000 resamples). Slim predicts c > 0 and smallest-eps slope approaching 1.
NOT SLIM if the pooled c has a 95% CI containing 0 (or below 0) and every smallest-eps local slope is << 1 (< 0.5)."""
import numpy as np

from common import load_eps_runs

rng = np.random.default_rng(0)
jobs = load_eps_runs()
rows = sorted({j["row"] for j in jobs})
def slopes(js, diff):
    f = [d.mean() for d in diff]
    ok = [i for i, x in enumerate(f) if 0.03 < x < 0.97]
    if len(ok) < 3: return None
    e = np.log([js[i]["eps"] for i in ok]); lf = np.log([f[i] for i in ok])
    return np.diff(lf) / np.diff(e)            # eps sorted descending -> positive slope = f falls with eps
cs, boots = {}, []
for r in rows:
    js = sorted([j for j in jobs if j["row"] == r], key=lambda j: -j["eps"])
    diff = [np.array(j["Ta"]) != np.array(j["Tb"]) for j in js]
    s = slopes(js, diff)
    if s is None: continue
    bs = []
    for _ in range(2000):
        bd = [d[rng.integers(0, len(d), len(d))] for d in diff]
        sb = slopes(js, bd)
        if sb is not None and len(sb) == len(s): bs.append(sb[-1] - sb[0])
    bs = np.array(bs)
    cs[r] = (s, s[-1] - s[0], bs)
    print(f"row {r:>5}: local slopes (large->small eps) {np.round(s, 3)}  curvature {s[-1]-s[0]:+.3f}  "
          f"95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
# pooled: mean curvature, bootstrap by combining per-puzzle bootstrap draws
n = min(len(v[2]) for v in cs.values())
pool = np.mean([v[2][:n] for v in cs.values()], axis=0)
c0 = np.mean([v[1] for v in cs.values()])
print(f"\npooled mean curvature {c0:+.3f}, 95% CI [{np.percentile(pool, 2.5):+.3f}, {np.percentile(pool, 97.5):+.3f}] over {len(cs)} puzzles")
print("largest smallest-eps local slope:", round(max(v[0][-1] for v in cs.values()), 3))
