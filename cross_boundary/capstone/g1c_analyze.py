"""Run the G1 analysis of g1c_tokens.py (unchanged gate) on the trained checkpoint minieqr_ckpt.pt."""
import json
import time

import numpy as np
import torch

import g1_tiny as G
import g1c_tokens as M

t0 = time.time()
ck = torch.load("minieqr_ckpt.pt")
m = M.MiniEqR()
m.load_state_dict(ck["model"])
m = m.double()
P_test, S_test = G.make_puzzles(300)
with torch.no_grad():
    x = m.inject(P_test).double()
    zH, zL = M.init(len(P_test), torch.float64)
    for _ in range(12):
        zH, zL = m.step(zH, zL, x)
    acc = float((m.logits(zH).argmax(-1).numpy() == S_test).all(1).mean())
print(f"steps trained {ck['done']}; test accuracy {acc:.3f}", flush=True)
rows = []
import os
START = int(os.environ.get("G1C_START", "0"))  # resume: analyse puzzles START..5 of the same fixed list
for i in np.argsort((P_test >= 0).sum(1))[START:6]:
    T, (A, B, q) = M.settle_map(m, P_test[i])
    kap, r2, nfit = G.kappa_fit(T)
    lams = []
    for j in np.argsort(T)[::-1][:8]:
        z0 = ((A[j] * q[:, 0] + B[j] * q[:, 1]) * np.sqrt(16 * M.d)).view(1, 16, M.d)
        loc = M.lyap1(m, P_test[i], z0, int(min(max(T[j], 20), 150)))
        lams.append(float(np.mean(loc[5:])))
    lam = float(np.median(lams))
    ok = bool(kap is not None and r2 >= 0.97 and lam > 0.1 and lam > kap)
    rows.append({"clues": int((P_test[i] >= 0).sum()), "T_max": int(T.max()), "T_median": float(np.median(T)),
                 "kappa": kap, "r2": r2, "n_fit": nfit, "lambda1": lam, "lambda1_all": np.round(lams, 3).tolist(),
                 "pass": ok})
    print("   ", rows[-1], f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
npass = sum(r["pass"] for r in rows)
res = {"steps": ck["done"], "test_acc": acc, "puzzles": rows, "n_pass": npass, "G1": bool(acc >= 0.9 and npass >= 2)}
print("G1:", res["G1"], f"({npass}/6 pass)")
json.dump(res, open(f"g1c_tokens_from{START}.json", "w"), indent=1)
