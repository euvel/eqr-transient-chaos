"""G1 attempt 2: competing solutions. Same tiny model and analysis as g1_tiny.py, but trained on 4x4 Sudoku puzzles
with 0-4 clues (usually several valid completions) with loss = soft-min over all valid completions of the
cross-entropy (loops 4..12), so the model may settle on ANY valid solution: several attractors per puzzle (B&L's
multistable setting). Analysis adds the outcome = which solution the map pixel ends in.
Gate G1 (fixed before the run): as in g1_tiny.py (exponential survival R^2 >= 0.97 over >= 8 loops and median
lambda_1 > max(0.1, kappa) on >= 2 of 6 test puzzles), counting only models whose final grids are valid solutions for
>= 90% of random starts.
"""
import json
import time

import numpy as np
import torch

import g1_tiny as G

rng = np.random.default_rng(1)


def make_multi(n, lo=0, hi=4):
    P, M = [], []
    while len(P) < n:
        s = G.SOL[rng.integers(len(G.SOL))]
        k = rng.integers(lo, hi + 1)
        idx = rng.choice(16, k, replace=False)
        c = -np.ones(16, int)
        c[idx] = s[idx]
        mask = np.all((G.SOL == c) | (c < 0), axis=1)
        P.append(c)
        M.append(mask)
    return np.array(P), np.array(M)


POOL_P, POOL_M = make_multi(20000)
SOLT = torch.tensor(G.SOL)                                      # (288, 16)


def loss_fn(logits, mask):
    lp = torch.log_softmax(logits, -1)                         # (B, 16, 4)
    ce = -lp.gather(2, SOLT.T[None].expand(len(lp), -1, -1)).sum(1)   # (B, 288): CE against every solution
    ce = ce.masked_fill(~mask, 1e9)
    return -torch.logsumexp(-ce, 1).mean()                     # soft-min over the valid completions


def train(D, Hd, steps=4000, seed=0):
    torch.manual_seed(seed)
    m = G.Tiny(D, Hd)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(steps):
        i = rng.integers(0, len(POOL_P), 256)
        x = G.encode(POOL_P[i]).float()
        mask = torch.tensor(POOL_M[i])
        z = torch.randn(len(i), D)
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True))
        loss = 0
        for t in range(12):
            z = m.step(z, x)
            if t >= 3:
                loss = loss + loss_fn(m.logits(z), mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def validity(m, D, P, n=200):
    x = G.encode(P[:n])
    z = torch.randn(n, D, dtype=torch.float64)
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True))
    for _ in range(40):
        z = m.step(z, x)
    g = m.logits(z).argmax(-1).numpy()
    ok = [(g[j] == G.SOL).all(1).any() and np.all((g[j] == P[j]) | (P[j] < 0)) for j in range(n)]
    return float(np.mean(ok))


if __name__ == "__main__":
    Ptest, Mtest = make_multi(300, 0, 3)
    sel = np.where(Mtest.sum(1) >= 2)[0]
    out = {}
    for D, Hd in ((32, 128), (64, 256)):
        t0 = time.time()
        m = train(D, Hd).double()
        val = validity(m, D, Ptest[sel])
        rows = []
        for i in sel[:6]:
            T, zf, (A, B, q) = G.settle_map(m, Ptest[i], D)
            g = m.logits(zf).argmax(-1).numpy()
            nsol = len({tuple(r) for r in g})
            kap, r2, nfit = G.kappa_fit(T)
            lams = []
            for j in np.argsort(T)[::-1][:8]:
                z0 = (A[j] * q[:, 0] + B[j] * q[:, 1]) * np.sqrt(D)
                loc = G.lyap1(m, Ptest[i], z0[None], int(min(max(T[j], 20), 200)))
                lams.append(float(np.mean(loc[5:])))
            lam = float(np.median(lams))
            ok = bool(kap is not None and r2 >= 0.97 and lam > 0.1 and lam > kap)
            rows.append({"clues": int((Ptest[i] >= 0).sum()), "n_valid_solutions": int(Mtest[i].sum()),
                         "distinct_outcomes_in_map": nsol, "T_max": int(T.max()), "T_median": float(np.median(T)),
                         "kappa": kap, "r2": r2, "n_fit": nfit, "lambda1": lam, "pass": ok})
        npass = sum(r["pass"] for r in rows)
        out[f"D{D}_H{Hd}"] = {"valid_frac": val, "puzzles": rows, "G1": bool(val >= 0.9 and npass >= 2),
                              "minutes": round((time.time() - t0) / 60, 1)}
        torch.save(m.state_dict(), f"multi_D{D}_H{Hd}.pt")
        print(f"D{D} H{Hd}: valid {val:.3f}, pass {npass}/6, G1 {out[f'D{D}_H{Hd}']['G1']}")
        for r in rows:
            print("   ", r, flush=True)
    json.dump(out, open("g1b_multi.json", "w"), indent=1)
