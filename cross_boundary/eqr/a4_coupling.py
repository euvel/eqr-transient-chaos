"""A4: is churn-drives-decision built into EqR's architecture? Token-group coupling via exact JVPs (laptop, float64).

At states on stored saddle trajectories (206 and 80, loops 20 and 40), perturb a token group S (random direction in
z_H and z_L of those tokens) and measure the leakage = ||(J^n v) outside S|| / ||J^n v|| after n = 1 and 3 loops.
Also z_H-only vs z_L-only perturbations. Groups: prefix (puzzle-embedding) tokens, contested cells (decoded value
changes on the trajectory), settled cells (never change), one contested cell, and random groups of matched size.
Prediction (written before the run): no group is decoupled; leakage of every group >= 0.3 after 1 loop and close to
its random-group baseline. A group with leakage < 0.01 after 3 loops would be an architectural invariant subspace.
"""
import json

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from eqr_common import load64, make_batch

s = load64()
m = s.model
SEQ = {"cos_sin": m.inner._cos_sin()}
PE = m.inner.config.puzzle_emb_len
rng = np.random.default_rng(0)


def jvp_n(row, h, l, vh, vl, n):
    b = make_batch(s, row, 1)
    X = m.inner._input_embeddings(b["inputs"], b["puzzle_identifiers"])
    with torch.no_grad():
        for _ in range(n):
            with fwAD.dual_level():
                h2, l2 = m.inner.deep_recursion(fwAD.make_dual(h, vh), fwAD.make_dual(l, vl), X, SEQ)
                vh, vl = fwAD.unpack_dual(h2).tangent, fwAD.unpack_dual(l2).tangent
                h, l = fwAD.unpack_dual(h2).primal, fwAD.unpack_dual(l2).primal
    return vh, vl


def leakage(vh, vl, toks):
    mask = torch.zeros(97, dtype=torch.bool)
    mask[toks] = True
    inside = vh[0, mask].square().sum() + vl[0, mask].square().sum()
    total = vh.square().sum() + vl.square().sum()
    return float(torch.sqrt((total - inside) / total))


def direction(toks, which="both"):
    vh = torch.zeros(1, 97, 512, dtype=torch.float64)
    vl = torch.zeros(1, 97, 512, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(rng.integers(1e9)))
    if which in ("both", "H"):
        vh[0, toks] = torch.randn(len(toks), 512, generator=g, dtype=torch.float64)
    if which in ("both", "L"):
        vl[0, toks] = torch.randn(len(toks), 512, generator=g, dtype=torch.float64)
    n = torch.sqrt(vh.square().sum() + vl.square().sum())
    return vh / n, vl / n


out = {"puzzle_emb_len": PE, "cases": []}
for row, f in ((206, "data/s2b_r206.npz"), (80, "data/s2b_r80.npz")):
    d = np.load(f)
    G = d["G"]
    changed = (G[1:] != G[:-1]).any(0)                      # per cell
    contested = [PE + c for c in np.where(changed)[0]]
    settled = [PE + c for c in np.where(~changed)[0]]
    for t in (20, 40):
        if t >= len(d["H"]):
            continue
        h = torch.from_numpy(d["H"][t])[None]
        l = torch.from_numpy(d["L"][t])[None]
        groups = {"prefix": list(range(PE)), "contested": contested, "settled": settled,
                  "one_contested": contested[:1]}
        for name, toks in list(groups.items()):
            if not toks:
                continue
            groups[f"random_{len(toks)}"] = sorted(rng.choice(97, len(toks), replace=False).tolist())
        for name, toks in groups.items():
            for n in (1, 3):
                vh, vl = jvp_n(row, h.clone(), l.clone(), *direction(toks), n)
                out["cases"].append({"row": row, "loop": t, "group": name, "size": len(toks), "n": n,
                                     "leak": round(leakage(vh, vl, toks), 4),
                                     "baseline_share_outside": round(1 - len(toks) / 97, 3)})
        for which in ("H", "L"):
            vh, vl = jvp_n(row, h.clone(), l.clone(), *direction(list(range(97)), which), 1)
            other = vl if which == "H" else vh
            frac = float(torch.sqrt(other.square().sum() / (vh.square().sum() + vl.square().sum())))
            out["cases"].append({"row": row, "loop": t, "group": f"z_{which}_only", "n": 1,
                                 "leak_into_other_latent": round(frac, 4)})
        print(f"row {row} loop {t}: {len(contested)} contested / {len(settled)} settled cells", flush=True)
for c in out["cases"]:
    print(c)
json.dump(out, open("a4_coupling.json", "w"), indent=1)
