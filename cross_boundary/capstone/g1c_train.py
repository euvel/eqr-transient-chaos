"""Resumable, time-guarded training for g1c_tokens.MiniEqR (same model, data and loss). Usage: g1c_train.py MAX_SECONDS"""
import os
import sys
import time

import numpy as np
import torch

import g1_tiny as G
import g1c_tokens as M

MAXS, TOTAL = float(sys.argv[1]), 3000
torch.manual_seed(0)
m = M.MiniEqR()
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
done = 0
if os.path.exists("minieqr_ckpt.pt"):
    ck = torch.load("minieqr_ckpt.pt")
    m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); done = ck["done"]
    G.rng = np.random.default_rng(1000 + done)
t0 = time.time()
while done < TOTAL and time.time() - t0 < MAXS:
    P, S = G.make_puzzles_cached(128)
    x = m.inject(P)
    zH, zL = M.init(len(P))
    loss = 0
    for t in range(12):
        zH, zL = m.step(zH, zL, x)
        if t >= 3:
            loss = loss + torch.nn.functional.cross_entropy(m.logits(zH).reshape(-1, 4), torch.tensor(S).reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    done += 1
    if done % 100 == 0:
        print(f"step {done}: loss {float(loss) / 9:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "done": done}, "minieqr_ckpt.pt")
torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "done": done}, "minieqr_ckpt.pt")
print("saved at step", done)
