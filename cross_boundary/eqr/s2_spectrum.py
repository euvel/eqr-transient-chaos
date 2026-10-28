"""S2: top-K Lyapunov spectrum (Benettin twins + QR, true float64) along slow EqR trajectories.

Usage: s2_spectrum.py ROW K CAP RANK OUT.npz   (RANK = 0 for the slowest pixel of the saved float32 map)
Saves per-loop local exponents (CAP, K), the base trajectory's decoded grids and its float64 settle time.
Twins: relative offset 1e-7 (float64), renormalised every loop; the state is (z_H, z_L) flattened.
"""
import sys
import time

import numpy as np
import torch

from eqr_common import load64, slowest_pixels, z_of_pixels, make_batch, suffix_settle, _prime_eqr_carry

row, K, CAP, rank, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
s = load64()
m = s.model
m.config.halt_max_steps = 10 ** 9
idx, T32 = slowest_pixels(row, rank + 1)
z0 = z_of_pixels(idx[rank:rank + 1])
B = K + 1
batch = make_batch(s, row, B)
carry = _prime_eqr_carry(m, batch, z_H=z0.repeat(B, 1, 1), z_L=z0.repeat(B, 1, 1) * 0)
ic = carry.inner_carry
nH, shH, shL = ic.z_H[0].numel(), ic.z_H.shape[1:], ic.z_L.shape[1:]


def state(ic):
    return torch.cat([ic.z_H.flatten(1), ic.z_L.flatten(1)], 1)


def put(ic, S0, Q, eps):
    tw = S0[None, :] + eps * Q.T
    ic.z_H[1:] = tw[:, :nH].reshape(K, *shH)
    ic.z_L[1:] = tw[:, nH:].reshape(K, *shL)


g = torch.Generator().manual_seed(1)
S = state(ic)
Q = torch.linalg.qr(torch.randn(S.shape[1], K, generator=g, dtype=torch.float64))[0]
eps = 1e-7 * float(S[0].norm().clamp(min=1.0))
put(ic, S[0], Q, eps)
local = np.zeros((CAP, K))
grids = []
t0 = time.time()
with torch.no_grad():
    for t in range(CAP):
        carry, out = m(carry, batch)
        carry.halted = torch.zeros_like(carry.halted)
        ic = carry.inner_carry
        S = state(ic)
        grids.append(np.asarray(s.task.decode_batch(out["logits"][:1]), dtype=np.int8).reshape(81))
        q, r = torch.linalg.qr((S[1:] - S[0][None, :]).T / eps)
        sg = torch.sign(torch.diagonal(r))
        Q = q * sg
        local[t] = torch.log(torch.abs(torch.diagonal(r))).numpy()
        eps = 1e-7 * float(S[0].norm().clamp(min=1.0))
        put(ic, S[0], Q, eps)
        if t % 10 == 9:
            Tset = int(suffix_settle(np.stack(grids)[None])[0])
            print(f"loop {t + 1}: {time.time() - t0:.0f}s, settle-so-far {Tset}, "
                  f"running means {np.round(local[:t + 1].mean(0)[:6], 3).tolist()} ...", flush=True)
            np.savez(OUT, local=local[:t + 1], grids=np.stack(grids), row=row, pixel=idx[rank], T32=T32[rank])
tr = np.stack(grids)
np.savez(OUT, local=local, grids=tr, row=row, pixel=idx[rank], T32=T32[rank], T64=suffix_settle(tr[None])[0])
print("done; float64 settle", suffix_settle(tr[None])[0], "float32 map settle", T32[rank])
