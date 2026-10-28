"""S3c: exact-JVP (forward-mode AD, float64) Benettin + Ginelli CLVs along a stored stagger trajectory.

Usage: s3c_clv_ad.py TRAJ.npz K MID OUT.npz [MAX_SECONDS]   (resumable: re-run with the same OUT to continue)
Processes loops [MID - 30, n - 1): 30 forward loops to converge Q, then stores Q at MID and every R matrix after it.
When all loops are done, Ginelli's backward iteration from the end to MID gives the CLVs at MID; a convergence check
repeats it from 15 loops before the end. Saves CLVs at MID (as K x state) and the local exponents.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from eqr_common import load64, make_batch

TRAJ, K, MID, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
MAXS = float(sys.argv[5]) if len(sys.argv) > 5 else 780
d = np.load(TRAJ)
row, Hs, Ls = int(d["row"]), torch.from_numpy(d["H"]), torch.from_numpy(d["L"])
n = len(Hs)
s = load64()
m = s.model
bK = make_batch(s, row, K)
XK = m.inner._input_embeddings(bK["inputs"], bK["puzzle_identifiers"])
SEQ = {"cos_sin": m.inner._cos_sin()}
nH = Hs[0].numel()
START = MID - 30


@torch.no_grad()
def jvp(t, Q):
    """J(S_t) @ Q for K tangent columns, exact (forward AD), batched."""
    h = Hs[t][None].expand(K, -1, -1).clone()
    l = Ls[t][None].expand(K, -1, -1).clone()
    th = Q[:nH].T.reshape(K, *Hs.shape[1:])
    tl = Q[nH:].T.reshape(K, *Ls.shape[1:])
    with fwAD.dual_level():
        h2, l2 = m.inner.deep_recursion(fwAD.make_dual(h, th), fwAD.make_dual(l, tl), XK, SEQ)
        jh, jl = fwAD.unpack_dual(h2).tangent, fwAD.unpack_dual(l2).tangent
    return torch.cat([jh.flatten(1), jl.flatten(1)], 1).T


if os.path.exists(OUT):
    st = np.load(OUT, allow_pickle=True)
    t0_loop, Q = int(st["next"]), torch.from_numpy(st["Q"])
    Rs, local = list(st["Rs"]), list(st["local"])
    QMID = torch.from_numpy(st["QMID"]) if st["QMID"].size > 1 else None
else:
    t0_loop = START
    Q = torch.linalg.qr(torch.randn(2 * nH, K, generator=torch.Generator().manual_seed(3), dtype=torch.float64))[0]
    Rs, local, QMID = [], [], None
T0 = time.time()
t = t0_loop
while t < n - 1 and time.time() - T0 < MAXS:
    if t == MID:
        QMID = Q.clone()
    q, r = torch.linalg.qr(jvp(t, Q))
    sg = torch.sign(torch.diagonal(r))
    Q, r = q * sg, r * sg[:, None]
    local.append(torch.log(torch.diagonal(r)).numpy())
    if t >= MID:
        Rs.append(r.numpy())
    t += 1
    if t % 10 == 0:
        print(f"loop {t}/{n - 1}: {time.time() - T0:.0f}s, mean exps {np.round(np.mean(local, 0)[:6], 2).tolist()}",
              flush=True)
        np.savez(OUT, next=t, Q=Q.numpy(), Rs=np.array(Rs), local=np.array(local),
                 QMID=QMID.numpy() if QMID is not None else np.zeros(1), row=row, MID=MID, K=K)
np.savez(OUT, next=t, Q=Q.numpy(), Rs=np.array(Rs), local=np.array(local),
         QMID=QMID.numpy() if QMID is not None else np.zeros(1), row=row, MID=MID, K=K)
if t < n - 1:
    print("time guard; resume with the same command", flush=True)
    sys.exit(0)


def ginelli(Rlist):
    C = np.eye(K)
    for R in Rlist[::-1]:
        C = np.linalg.solve(R, C)
        C /= np.linalg.norm(C, axis=0)
    return C


C_full, C_short = ginelli(Rs), ginelli(Rs[:-15])
V = QMID.numpy() @ C_full
V_short = QMID.numpy() @ C_short
conv = [float(min(np.linalg.norm(V[:, k] - V_short[:, k]), np.linalg.norm(V[:, k] + V_short[:, k]))) for k in range(K)]
lam = np.mean(local[30:], 0)  # exponents over the stored part (after the 30 warm-up loops)
np.savez(OUT, next=t, Q=Q.numpy(), Rs=np.array(Rs), local=np.array(local), QMID=QMID.numpy(), row=row, MID=MID, K=K,
         V=V, conv=np.array(conv), lam=lam)
print("done; exponents", np.round(lam, 3).tolist())
print("CLV convergence |V(full) - V(end-15)| per k:", np.array2string(np.array(conv), precision=1))
