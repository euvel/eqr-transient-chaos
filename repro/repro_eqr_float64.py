"""Minimal reproduction: EqR via loopscape is not float64 end-to-end (CPU, ~1 min).

(1) get_solver("eqr", dtype=torch.float64) leaves forward_dtype = float32, so the latent is injected and
    iterated in float32 (EqRSolver.load's dtype map has no float64 entry and falls back to "float32").
(2) Even with config_overrides={"forward_dtype": "float64"}, upstream EqR's rms_norm (models/layers.py)
    casts to float32 and back on every call, so trajectories are rounded to float32 each loop.
Test: two initial latents at relative distance eps; the separation after one loop should scale with eps.
"""
import sys
import torch
from loopscape import get_solver
from loopscape.eqr.solver import _prime_eqr_carry

PUZZLE = "..9..3.7.27..5.....3......8....1....7....6.2...65....49......41.6...2.3....4..8.."


@torch.no_grad()
def sep_after_one_loop(s, eps):
    m = s.model
    torch.manual_seed(0)
    z0 = torch.randn(1, 97, 512, dtype=torch.float64)
    w = torch.randn(1, 97, 512, dtype=torch.float64); w /= w.norm()
    z = torch.cat([z0, z0 + eps * z0.norm() * w])
    m.config.halt_max_steps = 1
    inp = s.task.encode(PUZZLE).repeat(2, 1)
    batch = {"inputs": inp, "labels": torch.full_like(inp, -100), "puzzle_identifiers": torch.zeros((2,), dtype=torch.int32)}
    carry, _ = m(_prime_eqr_carry(m, batch, z_H=z, z_L=z * 0), batch)
    zh = carry.inner_carry.z_H.flatten(1).double()
    return float((zh[1] - zh[0]).norm() / zh[0].norm())


def report(tag, s):
    r = sep_after_one_loop(s, 1e-6) / sep_after_one_loop(s, 1e-9)
    print(f"{tag:<46} forward_dtype={str(s.model.inner.forward_dtype):<14} d(1e-6)/d(1e-9) = {r:9.2f}  (float64 => ~1000)")


def noiseless(s):
    s.model.config.noise_scale = 0.0
    s.model.inner.L_level.noise_scale = 0.0
    return s


report('get_solver(dtype=float64)', noiseless(get_solver("eqr", device="cpu", dtype=torch.float64)))
s = noiseless(get_solver("eqr", device="cpu", dtype=torch.float64, config_overrides={"forward_dtype": "float64"}))
report('+ config_overrides forward_dtype="float64"', s)


def rms_norm_keep(h, variance_epsilon):           # same as loopscape/fprm/layers.py: upcast only fp16/bf16
    ct = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    x = h.to(ct)
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + variance_epsilon)).to(h.dtype)


for name in [n for n in list(sys.modules) if n.startswith("models.")]:
    if hasattr(sys.modules[name], "rms_norm"):
        setattr(sys.modules[name], "rms_norm", rms_norm_keep)
report('+ rms_norm patched to keep float64', s)
