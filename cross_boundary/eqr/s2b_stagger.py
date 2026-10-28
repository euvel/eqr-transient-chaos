"""S2b: long saddle trajectories of true-float64 EqR by stagger-and-step (Sweet, Nusse & Yorke, PRL 86, 2261, 2001).

Usage: s2b_stagger.py ROW NLOOPS OUT.npz [MAX_SECONDS] [RESUME.npz]
Starts at the slowest pixel of the saved map. Every ADV loops, look ahead TAU loops. The current state is kept if it is
still "alive": its decoded grid changes during the last 4 of the TAU look-ahead loops. Otherwise it is replaced by the
alive candidate with the smallest kick among B random kicks z + d*|z|*w, with d log-uniform in [1e-10, 1e-4] and w a
random unit vector in the full (z_H, z_L) state. Stores the full state at every loop (for S2c) and every kick.
"""
import os
import sys
import time

import numpy as np
import torch

from eqr_common import load64, slowest_pixels, z_of_pixels, make_batch, _prime_eqr_carry

row, NL, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MAXS = float(sys.argv[4]) if len(sys.argv) > 4 else 840
TAU, ADV, B = int(os.environ.get("STAGGER_TAU", 16)), int(os.environ.get("STAGGER_ADV", 6)), 12  # Amendment 2: look-ahead 16 (env override for S3)
s = load64()
m = s.model
m.config.halt_max_steps = 10 ** 9
g = torch.Generator().manual_seed(row)
batches = {}


def run(zH, zL, n):
    """Run n loops from states (b, 97, 512); return states after each loop and decoded grids."""
    b = zH.shape[0]
    if b not in batches:
        batches[b] = make_batch(s, row, b)
    carry = _prime_eqr_carry(m, batches[b], z_H=zH.clone(), z_L=zL.clone())
    H, L, G = [], [], []
    with torch.no_grad():
        for _ in range(n):
            carry, out = m(carry, batches[b])
            carry.halted = torch.zeros_like(carry.halted)
            H.append(carry.inner_carry.z_H.clone())
            L.append(carry.inner_carry.z_L.clone())
            G.append(np.asarray(s.task.decode_batch(out["logits"]), dtype=np.int8).reshape(b, 81))
    return torch.stack(H, 1), torch.stack(L, 1), np.stack(G, 1)  # (b, n, ...), (b, n, 81)


def alive(G, g_prev):
    """Decoded grid changes during the last 4 look-ahead loops."""
    full = np.concatenate([np.repeat(g_prev[None, None], G.shape[0], 0), G], 1)
    ch = (full[:, 1:] != full[:, :-1]).any(-1)
    return ch[:, -4:].any(1), np.where(ch, np.arange(1, ch.shape[1] + 1), 0).max(1)


import os
RANK = int(os.environ.get("STAGGER_RANK", "0"))  # 0 = slowest pixel; replicate runs use other ranks
idx, T32 = slowest_pixels(row, RANK + 1)
idx = idx[RANK:RANK + 1]
zH = z_of_pixels(idx)
zL = zH * 0
# exactness check: restarting from stored states reproduces a continuous run
H2, L2, _ = run(zH, zL, 2)
H1, L1, _ = run(H2[:, 0], L2[:, 0], 1)
assert torch.equal(H1[:, 0], H2[:, 1]) and torch.equal(L1[:, 0], L2[:, 1]), "restart is not exact"

if len(sys.argv) > 5:  # resume a stored trajectory (same rules; continues from its last state)
    rd = np.load(sys.argv[5])
    traj_H = list(torch.from_numpy(rd["H"]))
    traj_L = list(torch.from_numpy(rd["L"]))
    traj_G = list(rd["G"])
    kicks = [{"loop": int(a), "delta": float(b), "attempt": int(c)} for a, b, c in rd["kicks"]]
    g.manual_seed(row + 1000 * len(traj_H))
else:
    H0, L0, G0 = run(zH, zL, 1)
    traj_H, traj_L, traj_G = [H0[0, 0]], [L0[0, 0]], [G0[0, 0]]
    kicks = []
t0 = time.time()
while len(traj_H) < NL and time.time() - t0 < MAXS:
    cH, cL, gp = traj_H[-1][None], traj_L[-1][None], traj_G[-1]
    H, L, G = run(cH, cL, TAU)
    ok, last = alive(G, gp)
    if not ok[0]:
        chosen = None
        for attempt in range(4):  # Amendment 2
            S = torch.cat([cH.flatten(1), cL.flatten(1)], 1)
            W = torch.randn(B, S.shape[1], generator=g, dtype=torch.float64)
            W /= W.norm(dim=1, keepdim=True)
            d = 10 ** torch.empty(B, dtype=torch.float64).uniform_(-7, -2, generator=g)  # Amendment 2
            Sk = S + d[:, None] * S.norm() * W
            kH = Sk[:, :cH[0].numel()].reshape(B, *cH.shape[1:])
            kL = Sk[:, cH[0].numel():].reshape(B, *cL.shape[1:])
            H, L, G = run(kH, kL, TAU)
            ok, last = alive(G, gp)
            if ok.any():
                j = int(np.argmin(np.where(ok, d.numpy(), np.inf)))
                chosen = (j, float(d[j]), attempt)
                break
        if chosen is None:  # Amendment 2: backtrack ADV loops and kick from the earlier state
            if len(traj_H) > ADV + 1:
                del traj_H[-ADV:], traj_L[-ADV:], traj_G[-ADV:]
                kicks.append({"loop": len(traj_H), "delta": 0.0, "attempt": -2})  # -2 = backtrack marker
                print(f"backtrack to loop {len(traj_H)}", flush=True)
                continue
            j = int(np.argmax(last))
            chosen = (j, float(d[j]), -1)
        j = chosen[0]
        kicks.append({"loop": len(traj_H), "delta": chosen[1], "attempt": chosen[2]})
        # replace the current state by the kicked one; its future is H[j]
        traj_H[-1], traj_L[-1] = kH[j], kL[j]
        H, L, G = H[j:j + 1], L[j:j + 1], G[j:j + 1]
    for k in range(ADV):
        traj_H.append(H[0, k])
        traj_L.append(L[0, k])
        traj_G.append(G[0, k])
    print(f"loops {len(traj_H)}: {time.time() - t0:.0f}s, kicks {len(kicks)} "
          f"(backtracks {sum(k['attempt'] == -2 for k in kicks)}, failed {sum(k['attempt'] == -1 for k in kicks)})", flush=True)
    np.savez(OUT, H=torch.stack(traj_H).numpy(), L=torch.stack(traj_L).numpy(), G=np.stack(traj_G),
             kicks=np.array([[k["loop"], k["delta"], k["attempt"]] for k in kicks]).reshape(-1, 3), row=row,
             pixel=idx[0])
print("done", len(traj_H), "loops,", len(kicks), "kicks")
