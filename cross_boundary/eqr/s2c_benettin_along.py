"""S2c: top-K Lyapunov spectrum along a stored stagger-and-step trajectory (true float64, Benettin twins + QR).

Usage: s2c_benettin_along.py IN.npz K OUT.npz [MAX_SECONDS]
At each loop t the base is the stored state S_t; twins S_t + eps Q_k (eps = 1e-7 |S_t|); one loop; QR of the twin
differences against the base image F(S_t). Kicks (S_{t+1} != F(S_t)) only move the base; Q is carried on.
Checks that F(S_t) reproduces S_{t+1} exactly on every non-kick loop.
"""
import sys
import time

import numpy as np
import torch

from eqr_common import load64, make_batch, _prime_eqr_carry

IN, K, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MAXS = float(sys.argv[4]) if len(sys.argv) > 4 else 840
d = np.load(IN)
row, Hs, Ls = int(d["row"]), torch.from_numpy(d["H"]), torch.from_numpy(d["L"])
# the stagger log records len(traj) at kick time; the kicked state is the one before it
kick_loops = set(int(x) - 1 for x in d["kicks"][:, 0]) if len(d["kicks"]) else set()
n = len(Hs)
s = load64()
m = s.model
m.config.halt_max_steps = 10 ** 9
B = K + 1
batch = make_batch(s, row, B)
nH, shH, shL = Hs[0].numel(), Hs.shape[1:], Ls.shape[1:]
g = torch.Generator().manual_seed(1)
Q = torch.linalg.qr(torch.randn(nH + Ls[0].numel(), K, generator=g, dtype=torch.float64))[0]
local = np.full((n - 1, K), np.nan)
mismatch = []
t0 = time.time()
with torch.no_grad():
    for t in range(n - 1):
        if time.time() - t0 > MAXS:
            print("time guard at loop", t, flush=True)
            break
        S0 = torch.cat([Hs[t].flatten(), Ls[t].flatten()])
        eps = 1e-7 * float(S0.norm().clamp(min=1.0))
        tw = torch.cat([S0[None], S0[None] + eps * Q.T])
        carry = _prime_eqr_carry(m, batch, z_H=tw[:, :nH].reshape(B, *shH).clone(),
                                 z_L=tw[:, nH:].reshape(B, *shL).clone())
        carry, _ = m(carry, batch)
        S1 = torch.cat([carry.inner_carry.z_H.flatten(1), carry.inner_carry.z_L.flatten(1)], 1)
        if (t + 1) not in kick_loops:
            mismatch.append(float((S1[0] - torch.cat([Hs[t + 1].flatten(), Ls[t + 1].flatten()])).abs().max()))
        q, r = torch.linalg.qr((S1[1:] - S1[0][None]).T / eps)
        sg = torch.sign(torch.diagonal(r))
        Q = q * sg
        local[t] = torch.log(torch.abs(torch.diagonal(r))).numpy()
        if t % 10 == 9:
            print(f"loop {t + 1}/{n - 1}: {time.time() - t0:.0f}s, mean so far (from 6) "
                  f"{np.round(np.nanmean(local[6:t + 1], 0)[:8], 2).tolist() if t >= 6 else ''}", flush=True)
            np.savez(OUT, local=local, row=row, max_mismatch=max(mismatch) if mismatch else 0.0)
np.savez(OUT, local=local, row=row, max_mismatch=max(mismatch) if mismatch else 0.0)
print("done; max |F(S_t) - S_t+1| on non-kick loops:", max(mismatch) if mismatch else None)
