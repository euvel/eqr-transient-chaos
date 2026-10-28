"""Shared EqR harness for the cross-boundary smokes: true float64 (dtype map + rms_norm patched), map slice seed 0."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loopscape import get_solver
from loopscape.eqr.solver import _prime_eqr_carry

ROOT = Path(__file__).resolve().parents[2]  # repository root
EXP = ROOT / "experiments" / "01_settle_maps_and_lyapunov"
RESULTS = ROOT / "results"
QUESTIONS = {c["row"]: c["question"] for c in json.load(open(EXP / "candidates_v3.json"))["candidates"]}


def load64(threads=8):
    torch.set_num_threads(threads)
    s = get_solver("eqr", dtype=torch.float64, config_overrides={"forward_dtype": "float64"})
    s.model.config.noise_scale = 0.0
    s.model.inner.L_level.noise_scale = 0.0
    assert s.model.inner.forward_dtype == torch.float64

    def _rms_norm64(h, variance_epsilon):
        return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + variance_epsilon)
    for name in [n for n in list(sys.modules) if n.startswith("models.")]:
        if hasattr(sys.modules[name], "rms_norm"):
            setattr(sys.modules[name], "rms_norm", _rms_norm64)
    return s


def plane(seed=0):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(97 * 512, 2))
    return q[:, 0].reshape(97, 512), q[:, 1].reshape(97, 512)


def map_points(res=40):
    g = torch.linspace(-1, 1, res)
    x, y = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([x.ravel(), y.ravel()], -1)


def slowest_pixels(row, n):
    """Indices (into the 40x40 map) and float32 settle times of the n slowest pixels of a saved map."""
    for sub in ("maps_v4", "maps_v3b"):
        f = RESULTS / sub / f"main_r{row}.npz"
        if f.exists():
            T = np.load(f)["T_dec"]
            idx = np.argsort(T)[::-1][:n]
            return idx, T[idx]
    raise FileNotFoundError(row)


def z_of_pixels(idx):
    u, v = plane(0)
    p = map_points()[torch.as_tensor(np.array(idx))]
    return (p[:, 0, None, None] * u + p[:, 1, None, None] * v).double()


def make_batch(s, row, B):
    inputs = s.task.encode(QUESTIONS[row]).repeat(B, 1)
    return {"inputs": inputs, "labels": torch.full_like(inputs, -100),
            "puzzle_identifiers": torch.zeros((B,), dtype=torch.int32)}


def suffix_settle(tr):
    ch = (tr[:, 1:] != tr[:, :-1]).any(-1)
    return np.where(ch, np.arange(1, tr.shape[1]), 0).max(1)


__all__ = ["load64", "slowest_pixels", "z_of_pixels", "make_batch", "suffix_settle", "_prime_eqr_carry", "QUESTIONS"]
