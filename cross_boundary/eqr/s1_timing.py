"""S1: CPU timing of true-float64 EqR loops at batch sizes 1, 9, 17, 33 (K twins + base), puzzle 206 slowest pixel."""
import time

import torch

from eqr_common import load64, slowest_pixels, z_of_pixels, make_batch, _prime_eqr_carry

s = load64()
m = s.model
m.config.halt_max_steps = 10 ** 9
idx, T32 = slowest_pixels(206, 1)
z0 = z_of_pixels(idx)
print("pixel", idx, "float32 settle", T32)
for B in (1, 9, 17, 33):
    batch = make_batch(s, 206, B)
    carry = _prime_eqr_carry(m, batch, z_H=z0.repeat(B, 1, 1), z_L=z0.repeat(B, 1, 1) * 0)
    with torch.no_grad():
        carry, _ = m(carry, batch)  # warm-up
        t = time.time()
        for _ in range(3):
            carry, out = m(carry, batch)
            carry.halted = torch.zeros_like(carry.halted)
    dt = (time.time() - t) / 3
    print(f"B {B:3d}: {dt:.3f} s/loop  ({dt / B:.4f} s per sample-loop)", flush=True)
