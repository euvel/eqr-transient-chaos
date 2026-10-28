"""S3a: exact JVP of one EqR loop via forward-mode AD (float64) vs central differences."""
import time
import torch
import torch.autograd.forward_ad as fwAD
from eqr_common import load64, slowest_pixels, z_of_pixels, make_batch, _prime_eqr_carry

s = load64()
m = s.model
row = 1982
idx, _ = slowest_pixels(row, 1)
zH = z_of_pixels(idx)
zL = zH * 0
batch = make_batch(s, row, 1)


def step_ref(h, l):
    carry = _prime_eqr_carry(m, batch, z_H=h, z_L=l)
    carry, _ = m(carry, batch)
    return carry.inner_carry.z_H, carry.inner_carry.z_L


X = m.inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
SEQ = {"cos_sin": m.inner._cos_sin()}


def step(h, l):
    """One EqR loop via the model's own deep_recursion, without the final detach (keeps forward-AD tangents)."""
    return m.inner.deep_recursion(h, l, X, SEQ)


with torch.no_grad():
    h1, l1 = step(zH.clone(), zL.clone())      # move onto the trajectory first
    r1, q1 = step_ref(zH.clone(), zL.clone())
    print("step == model step (exact):", bool(torch.equal(h1, r1) and torch.equal(l1, q1)))
    g = torch.Generator().manual_seed(0)
    vH = torch.randn(h1.shape, generator=g, dtype=torch.float64)
    vL = torch.randn(l1.shape, generator=g, dtype=torch.float64)
    nv = torch.sqrt(vH.square().sum() + vL.square().sum())
    vH, vL = vH / nv, vL / nv
t = time.time()
with fwAD.dual_level(), torch.no_grad():
    h2, l2 = step(fwAD.make_dual(h1.clone(), vH), fwAD.make_dual(l1.clone(), vL))
    jH, jL = fwAD.unpack_dual(h2).tangent, fwAD.unpack_dual(l2).tangent
print("forward-AD JVP ok, %.1fs" % (time.time() - t))
with torch.no_grad():
    sc = torch.sqrt(h1.square().sum() + l1.square().sum())
    for e in (1e-3, 1e-5, 1e-7):
        hp, lp = step(h1 + e * sc * vH, l1 + e * sc * vL)
        hm, lm = step(h1 - e * sc * vH, l1 - e * sc * vL)
        dH, dL = (hp - hm) / (2 * e * sc), (lp - lm) / (2 * e * sc)
        err = torch.sqrt((dH - jH).square().sum() + (dL - jL).square().sum()) / torch.sqrt(jH.square().sum() + jL.square().sum())
        print(f"central diff eps {e:g}: relative difference to AD {float(err):.2e}")
