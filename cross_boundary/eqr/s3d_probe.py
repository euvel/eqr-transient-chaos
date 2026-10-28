"""S3d: direction-resolved lifetime probe at the CLV point of an EqR saddle trajectory (true float64).

Usage: s3d_probe.py TRAJ.npz CLV.npz KAPPA DIR OUT.json   (DIR = CLV index k >= 1, or 'rand')
From z = S_MID: follow its UNKICKED future to 8 loops before it settles (CLVs carried along with exact JVPs; they are
covariant), then PIM along z + s v (9 points per level, keep the neighbourhood of the longest-lived point). Lifetime =
loops until the decoded grid stops changing (suffix rule, 12 quiet loops). The slope dT/dln(1/w) is fitted over relative
widths w/|z| in [exp(-0.8 * 36.8 * kappa / lambda_1), 1e-1] (float64-noise window, S0b Amendment 1).
"""
import json
import sys
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from eqr_common import load64, make_batch

TRAJ, CLV, KAPPA, DIR, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
d, c = np.load(TRAJ), np.load(CLV)
row, MID = int(d["row"]), int(c["MID"])
V = torch.from_numpy(c["V"])
lam1 = float(c["lam"][0])
s = load64()
m = s.model
SEQ = {"cos_sin": m.inner._cos_sin()}
X = {}


def xemb(b):
    if b not in X:
        bt = make_batch(s, row, b)
        X[b] = m.inner._input_embeddings(bt["inputs"], bt["puzzle_identifiers"])
    return X[b]


@torch.no_grad()
def step(h, l):
    return m.inner.deep_recursion(h, l, xemb(h.shape[0]), SEQ)


@torch.no_grad()
def decode(h):
    logits = m.inner.lm_head(h[:, m.inner.config.puzzle_emb_len:])
    return np.asarray(s.task.decode_batch(logits), dtype=np.int8).reshape(h.shape[0], 81)


@torch.no_grad()
def lifetimes(h, l, g0, cap=80, quiet=12):
    """Loops until the decoded grid stops changing (relative to the previous grid g0), per batch row."""
    b = h.shape[0]
    last = np.zeros(b, int)
    prev = np.repeat(g0[None], b, 0)
    for t in range(1, cap + 1):
        h, l = step(h, l)
        g = decode(h)
        ch = (g != prev).any(1)
        last[ch] = t
        prev = g
        if t - last.max() >= quiet:
            break
    return last


H0, L0 = torch.from_numpy(d["H"][MID])[None], torch.from_numpy(d["L"][MID])[None]
g0 = d["G"][MID - 1] if MID > 0 else d["G"][MID]
nH = H0.numel()
T0 = time.time()
Trem = int(lifetimes(H0.clone(), L0.clone(), g0)[0])
# carry z and the CLVs forward along the unkicked future to 8 loops before settling
h, l, gprev, Vc = H0.clone(), L0.clone(), g0, V.clone()
K = V.shape[1]
for _ in range(max(0, Trem - 8)):
    hk, lk = h.expand(K, -1, -1).clone(), l.expand(K, -1, -1).clone()
    with torch.no_grad(), fwAD.dual_level():
        h2, l2 = m.inner.deep_recursion(fwAD.make_dual(hk, Vc[:nH].T.reshape(K, *h.shape[1:])),
                                        fwAD.make_dual(lk, Vc[nH:].T.reshape(K, *l.shape[1:])), xemb(K), SEQ)
        jh, jl = fwAD.unpack_dual(h2).tangent, fwAD.unpack_dual(l2).tangent
    Vc = torch.cat([jh.flatten(1), jl.flatten(1)], 1).T
    Vc = Vc / Vc.norm(dim=0, keepdim=True)
    gprev = decode(h)[0]
    h, l = step(h, l)
z = torch.cat([h.flatten(), l.flatten()])
zn = float(z.norm())
if DIR == "rand":
    v = torch.randn(z.shape, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    v /= v.norm()
else:
    v = Vc[:, int(DIR) - 1]
wmin = float(np.exp(-0.8 * 36.8 * KAPPA / lam1))
half, levels = 0.05, []
while True:  # adaptive half-width: the coarse line must show T variation
    s_ = torch.linspace(-half, half, 9, dtype=torch.float64) * zn
    P = z[None] + s_[:, None] * v[None]
    T = lifetimes(P[:, :nH].reshape(9, *h.shape[1:]), P[:, nH:].reshape(9, *l.shape[1:]), gprev)
    if np.ptp(T) > 0 or half >= 0.8:
        break
    half *= 4
a, b_ = -half * zn, half * zn
while True:
    s_ = torch.linspace(a, b_, 9, dtype=torch.float64)
    P = z[None] + s_[:, None] * v[None]
    T = lifetimes(P[:, :nH].reshape(9, *h.shape[1:]), P[:, nH:].reshape(9, *l.shape[1:]), gprev)
    i = int(np.argmax(T))
    levels.append({"width_rel": float((b_ - a) / zn), "Tmax": int(T[i]), "T": T.tolist()})
    print(f"width {(b_ - a) / zn:.1e}: T {T.tolist()}  [{time.time() - T0:.0f}s]", flush=True)
    a, b_ = float(s_[max(i - 1, 0)]), float(s_[min(i + 1, 8)])
    if (b_ - a) / zn < wmin / 4 or len(levels) >= 9:
        break
W = np.array([x["width_rel"] for x in levels])
TM = np.array([x["Tmax"] for x in levels])
msk = (W <= 0.1) & (W >= wmin)
slope = float(np.polyfit(np.log(1 / W[msk]), TM[msk], 1)[0]) if msk.sum() >= 3 and np.ptp(TM[msk]) > 0 else None
res = {"row": row, "dir": DIR, "lambda_dir": None if DIR == "rand" else float(c["lam"][int(DIR) - 1]),
       "lambda1": lam1, "kappa": KAPPA, "Trem_from_mid": Trem, "half_rel": half, "wmin_rel": wmin,
       "levels": levels, "slope": slope, "n_fit": int(msk.sum()), "seconds": round(time.time() - T0)}
json.dump(res, open(OUT, "w"), indent=1)
print("slope", slope, "over", int(msk.sum()), "levels; 1/lambda1 =", round(1 / lam1, 2),
      "; 1/lambda_dir =", None if DIR == "rand" else round(1 / res["lambda_dir"], 2))
