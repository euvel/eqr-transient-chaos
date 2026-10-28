"""G1 attempt 3 (last, with a hard stop): a faithful miniature of EqR.
16 cell tokens x d = 16; latents z_H, z_L (16 x 16 each); input x = embed(cell value) + position.
One loop = deep recursion as in EqR: H_CYC times { L_CYC times z_L = block(z_L + z_H + x); z_H = block(z_H + z_L) },
block = post-norm transformer block (2-head attention, no mask; SwiGLU MLP x4): h = rms(h + attn(h)); h = rms(h + mlp(h)).
Readout: logits = lm_head(z_H). Random latent starts; deep supervision (cross-entropy at loops 4..12); 4x4 Sudoku with a
unique solution, 4-8 clues.
Gate G1 (unchanged from attempts 1-2, fixed before the run): exponential survival (R^2 >= 0.97 over >= 8 loops) and median
lambda_1 on the 8 slowest trajectories > max(0.1, kappa), on >= 2 of 6 test puzzles; test accuracy >= 90%.
Maps: 2-D slice in z_H (z_L = 0), as in loopscape's EqR slices.
"""
import json
import time

import numpy as np
import torch

import g1_tiny as G

torch.set_num_threads(8)
d, H_CYC, L_CYC = 16, 2, 2


def rms(h):
    return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.o = torch.nn.Linear(d, d, bias=False)
        self.gu = torch.nn.Linear(d, 8 * d, bias=False)
        self.dn = torch.nn.Linear(4 * d, d, bias=False)

    def forward(self, h):
        B, N, _ = h.shape
        q, k, v = self.qkv(h).view(B, N, 3, 2, d // 2).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(d // 2), -1) @ v
        h = rms(h + self.o(a.transpose(1, 2).reshape(B, N, d)))
        g, u = self.gu(h).chunk(2, -1)
        return rms(h + self.dn(torch.nn.functional.silu(g) * u))


class MiniEqR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(5, d)
        self.pos = torch.nn.Parameter(torch.randn(16, d) * 0.1)
        self.blk = Block()
        self.head = torch.nn.Linear(d, 4)

    def inject(self, P):
        return self.emb(torch.as_tensor(P + 1)) + self.pos

    def step(self, zH, zL, x):
        for _ in range(H_CYC):
            for _ in range(L_CYC):
                zL = self.blk(zL + zH + x)
            zH = self.blk(zH + zL)
        return zH, zL

    def logits(self, zH):
        return self.head(zH)


def init(B, dtype=torch.float32):
    z = torch.randn(B, 16, d, dtype=dtype)
    return rms(z), rms(torch.randn(B, 16, d, dtype=dtype))


def train(steps=3000, seed=0):
    torch.manual_seed(seed)
    m = MiniEqR()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(steps):
        P, S = G.make_puzzles_cached(128)
        x = m.inject(P)
        zH, zL = init(len(P))
        loss = 0
        for t in range(12):
            zH, zL = m.step(zH, zL, x)
            if t >= 3:
                loss = loss + torch.nn.functional.cross_entropy(m.logits(zH).reshape(-1, 4), torch.tensor(S).reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def settle_map(m, p, res=64, cap=300, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(16 * d, 2, generator=g, dtype=torch.float64))
    a = torch.linspace(-1, 1, res, dtype=torch.float64)
    A, B = torch.meshgrid(a, a, indexing="ij")
    A, B = A.reshape(-1), B.reshape(-1)
    zH = ((A[:, None] * q[:, 0] + B[:, None] * q[:, 1]) * np.sqrt(16 * d)).view(-1, 16, d)
    zL = torch.zeros_like(zH)
    x = m.inject(p[None]).double()
    prev, last = None, torch.zeros(len(zH), dtype=torch.long)
    for t in range(1, cap + 1):
        zH, zL = m.step(zH, zL, x)
        gg = m.logits(zH).argmax(-1)
        if prev is not None:
            last[(gg != prev).any(-1)] = t
        prev = gg
    return last.numpy(), (A, B, q)


def lyap1(m, p, zH, n):
    x = m.inject(p[None]).double()
    zL = torch.zeros_like(zH)
    f = lambda a, b: m.step(a, b, x)
    vH, vL = torch.randn_like(zH), torch.randn_like(zL)
    nv = torch.sqrt(vH.square().sum() + vL.square().sum())
    vH, vL = vH / nv, vL / nv
    out = []
    for _ in range(n):
        (h2, l2), (jh, jl) = torch.autograd.functional.jvp(f, (zH, zL), (vH, vL))
        zH, zL = h2.detach(), l2.detach()
        nv = torch.sqrt(jh.square().sum() + jl.square().sum())
        out.append(float(torch.log(nv)))
        vH, vL = jh / nv, jl / nv
    return out


if __name__ == "__main__":
    t0 = time.time()
    P_test, S_test = G.make_puzzles(300)
    m = train()
    m = m.double()
    with torch.no_grad():
        x = m.inject(P_test).double()
        zH, zL = init(len(P_test), torch.float64)
        for _ in range(12):
            zH, zL = m.step(zH, zL, x)
        acc = float((m.logits(zH).argmax(-1).numpy() == S_test).all(1).mean())
    print(f"trained in {(time.time() - t0) / 60:.1f} min; test accuracy {acc:.3f}", flush=True)
    rows = []
    for i in np.argsort((P_test >= 0).sum(1))[:6]:
        T, (A, B, q) = settle_map(m, P_test[i])
        kap, r2, nfit = G.kappa_fit(T)
        lams = []
        for j in np.argsort(T)[::-1][:8]:
            z0 = ((A[j] * q[:, 0] + B[j] * q[:, 1]) * np.sqrt(16 * d)).view(1, 16, d)
            loc = lyap1(m, P_test[i], z0, int(min(max(T[j], 20), 150)))
            lams.append(float(np.mean(loc[5:])))
        lam = float(np.median(lams))
        ok = bool(kap is not None and r2 >= 0.97 and lam > 0.1 and lam > kap)
        rows.append({"clues": int((P_test[i] >= 0).sum()), "T_max": int(T.max()), "T_median": float(np.median(T)),
                     "kappa": kap, "r2": r2, "n_fit": nfit, "lambda1": lam, "pass": ok})
        print("   ", rows[-1], flush=True)
    npass = sum(r["pass"] for r in rows)
    res = {"test_acc": acc, "puzzles": rows, "n_pass": npass, "G1": bool(acc >= 0.9 and npass >= 2),
           "minutes": round((time.time() - t0) / 60, 1)}
    print("G1:", res["G1"], f"({npass}/6 pass)")
    torch.save(m.state_dict(), "minieqr.pt")
    json.dump(res, open("g1c_tokens.json", "w"), indent=1)
