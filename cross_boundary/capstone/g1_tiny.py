"""G1: does a TINY trained looped reasoner show transient chaos? (laptop CPU; capstone gate 1)

Task: 4x4 Sudoku (2x2 boxes; 288 solutions); puzzles with 4-8 clues and a unique solution.
Model (EqR's ingredients, minimal): latent z in R^D; z <- rmsnorm(z + W2 silu(W1 [z; x] + b1)); logits = R z (16 cells x 4).
Training: random starting latents, 12 unrolled loops, cross-entropy at loops 4..12 (deep supervision), Adam.
Analysis (float64): for test puzzles, settle-time maps (64x64) on a 2-D slice of starting latents, settle time = last loop
at which the decoded grid changes (cap 300); survival S(k) = P(T > k) with an exponential fit; lambda_1 by Benettin
(exact Jacobian via autograd) on the 8 slowest trajectories.
Gate G1 (fixed before the run): PASS for a model if, on at least 2 test puzzles, (a) the exponential fit of S over
0.9 >= S >= 0.01 has R^2 >= 0.97 across >= 8 loops and (b) the median lambda_1 on slow trajectories is > 0.1 and
> kappa. Test accuracy (from random starts, at loop 12) is reported; a model must solve >= 90% of test puzzles to count.
"""
import itertools
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(8)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- 4x4 Sudoku data
def solutions():
    out = []
    for g in itertools.product(*[list(itertools.permutations(range(4)))] * 4):
        G = np.array(g)
        if all(len(set(G[:, c])) == 4 for c in range(4)) and all(
                len(set(G[r:r + 2, c:c + 2].ravel())) == 4 for r in (0, 2) for c in (0, 2)):
            out.append(G.ravel())
    return np.array(out)


SOL = solutions()
assert len(SOL) == 288


def n_solutions(clues):  # clues: length-16 array, -1 = blank
    m = np.all((SOL == clues) | (clues < 0), axis=1)
    return int(m.sum()), SOL[m]


def make_puzzles(n, lo=4, hi=8):
    P, S = [], []
    while len(P) < n:
        s = SOL[rng.integers(len(SOL))]
        k = rng.integers(lo, hi + 1)
        idx = rng.choice(16, k, replace=False)
        c = -np.ones(16, int)
        c[idx] = s[idx]
        cnt, _ = n_solutions(c)
        if cnt == 1:
            P.append(c)
            S.append(s)
    return np.array(P), np.array(S)


def encode(P):
    x = np.zeros((len(P), 16, 5))
    x[np.arange(len(P))[:, None], np.arange(16)[None], P + 1] = 1
    return torch.tensor(x.reshape(len(P), 80))


# ---------------------------------------------------------------- model
class Tiny(torch.nn.Module):
    def __init__(self, D, Hd):
        super().__init__()
        self.D = D
        self.W1 = torch.nn.Linear(D + 80, Hd)
        self.W2 = torch.nn.Linear(Hd, D)
        self.R = torch.nn.Linear(D, 64)

    def step(self, z, x):
        h = z + self.W2(torch.nn.functional.silu(self.W1(torch.cat([z, x], -1))))
        return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)

    def logits(self, z):
        return self.R(z).view(-1, 16, 4)


def train(D, Hd, steps=4000, seed=0):
    torch.manual_seed(seed)
    m = Tiny(D, Hd)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for it in range(steps):
        P, S = make_puzzles_cached(256)
        x = encode(P).float()
        z = torch.randn(len(P), D)
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True))
        loss = 0
        for t in range(12):
            z = m.step(z, x)
            if t >= 3:
                loss = loss + torch.nn.functional.cross_entropy(m.logits(z).reshape(-1, 4), torch.tensor(S).reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


POOL_P, POOL_S = make_puzzles(20000)


def make_puzzles_cached(n):
    i = rng.integers(0, len(POOL_P), n)
    return POOL_P[i], POOL_S[i]


# ---------------------------------------------------------------- analysis (float64)
@torch.no_grad()
def settle_map(m, p, D, res=64, cap=300, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D, 2, generator=g, dtype=torch.float64))
    a = torch.linspace(-1, 1, res, dtype=torch.float64)
    A, B = torch.meshgrid(a, a, indexing="ij")
    z = (A.reshape(-1, 1) * q[:, 0] + B.reshape(-1, 1) * q[:, 1]) * np.sqrt(D)
    x = encode(p[None]).repeat(len(z), 1)
    prev = None
    last = torch.zeros(len(z), dtype=torch.long)
    for t in range(1, cap + 1):
        z = m.step(z, x)
        g_ = m.logits(z).argmax(-1)
        if prev is not None:
            ch = (g_ != prev).any(-1)
            last[ch] = t
        prev = g_
    return last.numpy(), z, (A.reshape(-1), B.reshape(-1), q)


def kappa_fit(T, cap=300):
    S = np.array([(T > k).mean() for k in range(cap)])
    msk = (S <= 0.9) & (S >= 0.01)
    if msk.sum() < 8:
        return None, None, int(msk.sum())
    k = np.arange(cap)[msk]
    c = np.polyfit(k, np.log(S[msk]), 1)
    pred = np.polyval(c, k)
    r2 = 1 - np.sum((np.log(S[msk]) - pred) ** 2) / np.sum((np.log(S[msk]) - np.log(S[msk]).mean()) ** 2)
    return float(-c[0]), float(r2), int(msk.sum())


def lyap1(m, p, z0, n_loops):
    x = encode(p[None])
    z = z0.clone()
    v = torch.randn_like(z)
    v /= v.norm()
    s = []
    for _ in range(n_loops):
        _, jv = torch.autograd.functional.jvp(lambda zz: m.step(zz, x), (z,), (v,))
        with torch.no_grad():
            z = m.step(z, x)
        nv = jv.norm()
        s.append(float(torch.log(nv)))
        v = jv / nv
    return s


def analyse(m, D, P_test, S_test, n_puzzles=6):
    m = m.double()
    with torch.no_grad():
        x = encode(P_test)
        z = torch.randn(len(P_test), D, dtype=torch.float64)
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True))
        for _ in range(12):
            z = m.step(z, x)
        acc = float((m.logits(z).argmax(-1).numpy() == S_test).all(1).mean())
    rows = []
    order = np.argsort((P_test >= 0).sum(1))[:n_puzzles * 3]   # fewest clues first
    for i in order[:n_puzzles]:
        T, _, (A, B, q) = settle_map(m, P_test[i], D)
        kap, r2, nfit = kappa_fit(T)
        slow = np.argsort(T)[::-1][:8]
        lams = []
        for j in slow:
            z0 = (A[j] * q[:, 0] + B[j] * q[:, 1]) * np.sqrt(D)
            loc = lyap1(m, P_test[i], z0[None], int(min(T[j], 200)))
            if len(loc) > 10:
                lams.append(float(np.mean(loc[5:])))
        lam = float(np.median(lams)) if lams else None
        ok = bool(kap is not None and r2 is not None and r2 >= 0.97 and lam is not None and lam > 0.1 and lam > kap)
        rows.append({"clues": int((P_test[i] >= 0).sum()), "T_max": int(T.max()), "T_median": float(np.median(T)),
                     "kappa": kap, "r2": r2, "n_fit": nfit, "lambda1": lam, "pass": ok})
    return acc, rows


if __name__ == "__main__":
    P_test, S_test = make_puzzles(300)
    out = {}
    for D, Hd in ((32, 128), (64, 256)):
        t0 = time.time()
        m = train(D, Hd)
        acc, rows = analyse(m, D, P_test, S_test)
        npass = sum(r["pass"] for r in rows)
        out[f"D{D}_H{Hd}"] = {"test_acc": acc, "puzzles": rows, "n_pass": npass,
                              "G1": bool(acc >= 0.9 and npass >= 2), "minutes": round((time.time() - t0) / 60, 1)}
        torch.save(m.state_dict(), f"tiny_D{D}_H{Hd}.pt")
        print(f"D{D} H{Hd}: acc {acc:.3f}, pass {npass}/6, G1 {out[f'D{D}_H{Hd}']['G1']} [{out[f'D{D}_H{Hd}']['minutes']} min]")
        for r in rows:
            print("   ", r, flush=True)
    json.dump(out, open("g1_tiny.json", "w"), indent=1)
