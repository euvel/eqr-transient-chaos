"""v3b: boundary roughness H vs kappa/lambda1 on EqR (GilpinLab/loopscape). PREREG_v3.md + Amendment 1."""

import io
import json
import os
import pickletools
import subprocess
import sys
import threading
import time
import traceback
import zipfile

T0 = time.time()
ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = "/kaggle/working/out" if ON_KAGGLE else "out"
os.makedirs(OUT, exist_ok=True)
GUARD_S = 10 * 3600
SMALL = os.environ.get("EQR_SMALL") == "1"
LOG = open(f"{OUT}/log.txt", "a")
LOCK = threading.Lock()


def log(*a):
    msg = f"[{(time.time() - T0) / 60:7.1f} min] " + " ".join(str(x) for x in a)
    with LOCK:
        print(msg, flush=True)
        LOG.write(msg + "\n")
        LOG.flush()


if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "loopscape @ git+https://github.com/GilpinLab/loopscape"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "omegaconf", "einops"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from loopscape import fractal as F  # noqa: E402
from loopscape import get_solver  # noqa: E402
from loopscape.eqr.download import ensure_eqr_checkpoint  # noqa: E402
from loopscape.eqr.solver import _prime_eqr_carry  # noqa: E402

HARD = "..4.6...16..3...8.....24.....9..2..4.5.......4...3.26.....8...........172....384."
EASY = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."
P = {  # PREREG_v2: Sudoku-Extreme test.csv rows 81, 470, 1214 (ratings 112, 110, 113)
    "P1": "4..5.19..3..........5.6..4.7.....8....12...5..3......9..6..5.2.....4...8.7.62....",
    "P2": ".27.8..3.9..3.............16..42...8...87.....48..3.2....2...4.5.....6...91......",
    "P3": "4.9...7...1......8....9..4..3..2.4..1....4..38..3......4.1....2....765..5...4..6.",
    "HARD": HARD, "EASY": EASY,
}
TAU = 0.05
NDEV = torch.cuda.device_count() if torch.cuda.is_available() else 1
DEVS = [f"cuda:{i}" for i in range(NDEV)] if torch.cuda.is_available() else ["cpu"]
REPORT = {"prereg": "PREREG_v3.md + Amendment 1", "devices": [torch.cuda.get_device_name(i) for i in range(NDEV)]
          if torch.cuda.is_available() else ["cpu"], "torch": torch.__version__, "maps": {}, "checks": {}}

RES_S, CAP_S, RES, CAP, BATCH, N_LY, K_LY, CAP_LY, MAX_SEL = 16, 60, 40, 150, 256, 8, 4, 100, 8
MED_LO, MED_HI, MAX_CENS, MIN_SPAN = 2, 30, 0.10, 3
if SMALL:
    RES_S, CAP_S, RES, CAP, BATCH, N_LY, K_LY, CAP_LY, MAX_SEL = 3, 10, 4, 12, 16, 2, 2, 8, 3
    MED_LO, MED_HI, MAX_CENS = 0, 50, 1.0


def save_report():
    with LOCK:
        with open(f"{OUT}/report.json", "w") as f:
            json.dump(REPORT, f, indent=1, default=float)


# ---------------------------------------------------------------- checkpoint audit
ALLOWED = {"__builtin__ dict", "__builtin__ list", "__builtin__ long", "collections OrderedDict",
           "collections defaultdict", "typing Any", "torch._utils _rebuild_tensor_v2", "torch FloatStorage"}


def audit(path):
    refs = set()
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("data.pkl"):
                strings = []
                for op, arg, _ in pickletools.genops(io.BytesIO(z.read(n))):
                    if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                        strings.append(arg)
                    elif op.name == "GLOBAL":
                        refs.add(arg)
                    elif op.name == "STACK_GLOBAL":
                        refs.add(f"{strings[-2]} {strings[-1]}")
    return sorted(r for r in refs if r not in ALLOWED and not r.startswith("omegaconf."))


bad = audit(ensure_eqr_checkpoint(task="sudoku"))
if bad:
    REPORT["abort"] = f"unexpected pickle globals {bad}"
    save_report()
    raise SystemExit(REPORT["abort"])
log("checkpoint audit ok; devices", REPORT["devices"])


# ---------------------------------------------------------------- model + stepping
def load(dtype, device):
    over = {"forward_dtype": "float64"} if dtype == torch.float64 else None
    s = get_solver("eqr", device=device, dtype=dtype, config_overrides=over)
    s.model.config.noise_scale = 0.0
    s.model.inner.L_level.noise_scale = 0.0
    assert s.model.inner.forward_dtype == dtype
    return s


def plane(seed):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(97 * 512, 2))
    return q[:, 0].reshape(97, 512), q[:, 1].reshape(97, 512)


def mesh(res):
    g = torch.linspace(-1, 1, res)
    x, y = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([x.ravel(), y.ravel()], -1)


def latents(seed, res, sub=None):
    u, v = plane(seed)
    p = mesh(res)
    if sub is not None:
        p = p[sub]
    return p[:, 0, None, None] * u + p[:, 1, None, None] * v


def suffix_settle(tr):
    """tr (N, cap, 81): loops until the decoded grid never changes again; cap-1 = still changing at the end."""
    ch = (tr[:, 1:] != tr[:, :-1]).any(-1)
    return np.where(ch, np.arange(1, tr.shape[1]), 0).max(1)


@torch.no_grad()
def run_batch(solver, puzzle, zb, cap):
    """Same loop as EqRSolver.solve_batch (latent-sweep mode), plus the per-step zH residual."""
    m = solver.model
    dev = next(m.parameters()).device
    m.config.halt_max_steps = int(cap)
    B = zb.shape[0]
    inputs = solver.task.encode(puzzle).to(dev).repeat(B, 1)
    batch = {"inputs": inputs, "labels": torch.full_like(inputs, -100),
             "puzzle_identifiers": torch.zeros((B,), dtype=torch.int32, device=dev)}
    carry = _prime_eqr_carry(m, batch, z_H=zb, z_L=zb * 0)
    prev = carry.inner_carry.z_H.clone()
    grids, res = [], []
    steps = 0
    while True:
        carry, out = m(carry, batch)
        steps += 1
        grids.append(np.asarray(solver.task.decode_batch(out["logits"]), dtype=np.int8).reshape(B, 81))
        zh = carry.inner_carry.z_H
        r = (prev - zh).abs().amax(dim=(1, 2)) / (zh.abs().amax(dim=(1, 2)) + 1e-30)
        res.append(r.double().cpu().numpy())
        prev = zh.clone()
        if bool(carry.halted.all()) or steps >= cap:
            break
    return np.stack(grids, 1), np.stack(res, 1)          # (B, cap, 81), (B, cap)


def run_map(solver, puzzle, z, cap, batch, label):
    trs, rs = [], []
    t = time.time()
    for i in range(0, len(z), batch):
        tr, r = run_batch(solver, puzzle, z[i:i + batch], cap)
        trs.append(tr)
        rs.append(r)
        if i == 0:
            rate = len(tr) * cap / (time.time() - t)
            log(f"  {label}: {rate:.1f} sample-loops/s, eta {len(z) * cap / rate / 60:.1f} min")
    tr, r = np.concatenate(trs), np.concatenate(rs)
    T_dec = suffix_settle(tr)
    under = r < TAU
    T_tau = np.where(under.any(1), under.argmax(1), cap)
    final = tr[:, -1]
    solved = []
    for g in final:
        try:
            solved.append(bool(solver.task.is_solved(puzzle, g.reshape(9, 9))))
        except Exception:
            solved.append(bool(solver.task.is_solved(puzzle, g.reshape(9, 9).tolist())))
    solved = np.array(solved)
    return T_dec, T_tau, solved


# ---------------------------------------------------------------- statistics
def escape(T, cap, n_boot=200):
    def fit(s):
        n = np.arange(cap - 1)
        S = np.array([(s > k).mean() for k in n])
        if not (S <= 0.9).any():
            return None
        a = int(np.argmax(S <= 0.9))
        ok = np.where(S >= 0.02)[0]
        if len(ok) == 0:
            return None
        b = int(min(cap - 5, ok.max()))
        if b - a < MIN_SPAN or S[b] <= 0 or S[a] / S[b] < np.e:
            return {"range": [a, b], "too_short": True}
        x, y = n[a:b + 1], np.log(S[a:b + 1])
        k, c = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (k * x + c)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return {"kappa": float(-k), "r2": float(r2), "range": [a, b], "S_range": [float(S[a]), float(S[b])]}

    main = fit(T)
    if not main or "kappa" not in main:
        return {"fit": main}
    rng = np.random.default_rng(0)
    ks = [f["kappa"] for f in (fit(T[rng.integers(0, len(T), len(T))]) for _ in range(n_boot))
          if f and "kappa" in f]
    main["kappa_ci95"] = [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))] if ks else None
    return main


def lab_stats(T, n_max, res):
    try:
        s = F.field_stats(np.asarray(T), n_max, res, F.smooth_null(res))
        keep = {k: s[k] for k in ("H", "H_raw", "alpha", "alpha_r2", "D", "saturation", "r2",
                                  "frac_censored", "n_distinct_T", "n_components")}
        keep["smooth_null_D"] = [min(s["smooth_null"]), max(s["smooth_null"])] if s["smooth_null"] else None
        return keep
    except Exception as e:
        return {"error": repr(e)}


def summarize(T_dec, T_tau, solved, cap, res):
    out = {"cap": cap, "res": res, "n": int(len(T_dec)),
           "solved_frac": float(solved.mean()),
           "dec_censored_frac": float((T_dec >= cap - 1).mean()),
           "tau_censored_frac": float((T_tau >= cap).mean()),
           "dec_censored_at24": float((T_dec >= 23).mean()) if cap > 24 else None,
           "escape_dec": escape(T_dec, cap)}
    out["paper_inclusion_rule_pass"] = bool(out["solved_frac"] >= 0.9 and out["dec_censored_frac"] <= 0.01)
    out["lab_dec_full"] = lab_stats(T_dec.reshape(res, res), cap - 1, res)
    out["lab_tau_full"] = lab_stats(T_tau.reshape(res, res), cap, res)
    if cap > 24:
        tr24 = np.minimum(T_dec, 23)
        out["lab_dec_cap24"] = lab_stats(tr24.reshape(res, res), 23, res)
    return out


CANDS = [{"row": 3, "rating": 60, "q": ".....8..1.6.9..2...4.52.9..4..6......9..45..6....924..9......7........3.35.2..6.."}, {"row": 10, "rating": 50, "q": "4..2....17..6.5..4..9....6......65......8....3..7....2.7......32.....14..3.1...2."}, {"row": 36, "rating": 60, "q": "..6.....2...6.95..5...2..3.7.9.6.2...1.5.......5.9.3...7.....4...3.7.9.......8..3"}, {"row": 37, "rating": 56, "q": "..7..561..6.......1...2....7.....4...8...157.....4...96....485...85....7..53....."}, {"row": 38, "rating": 68, "q": ".84..5.6.....8.7..3...4.....384...2...6..3..89.......6...5..........2..1.25.3..8."}, {"row": 47, "rating": 54, "q": ".27..9...1...4.2...4.23...5.....3.5...142.7.....9....6.1..7.8..7......6.8..3....."}, {"row": 50, "rating": 53, "q": "3....98....8.3..1.7..8..3.52..9.......5.8...1.7...6.....4.5.1...2.....58...6...4."}, {"row": 52, "rating": 50, "q": "7......46.52...7...6...7.5...1.8..6.5....64.....1.4..8.3.9.......5.3..9.2....56.."}, {"row": 74, "rating": 74, "q": "8..3..6...7.1.8...3...5..1...64..9...2.......1....5.4.2..5...8......3..9..7.2.4.."}, {"row": 77, "rating": 62, "q": "8...5.7.47..1...5...5..4..9.6.........7.8..4.4....2....5..4.67.3.....4...7..1..9."}, {"row": 80, "rating": 76, "q": ".7.1....9..4....8.3...897.......2.........24..3.7....6.1.....9.6.39.......7.51..3"}, {"row": 102, "rating": 58, "q": ".9..8......53...9.43...75....6...1...5.6...3.3....82..56.4...1......1...9..87..4."}, {"row": 121, "rating": 54, "q": "586.3.2..3...2.85.........4.63.1...8..1...6..8....2...6...5.1...9......5...7....."}, {"row": 131, "rating": 55, "q": "..8...2..5....3....7..9..45...2..1.......8....6..4..9..5.....7...13.....4...6...9"}, {"row": 149, "rating": 84, "q": "8....6....1..4...63.....7...5.2....47....4.......5.9...2...5..1..3.1..8.....2..9."}, {"row": 169, "rating": 54, "q": "4..5...6...1.8............3.4.....9...94..6.55....9..7.2...5...7..9...5.9.56...74"}, {"row": 172, "rating": 95, "q": ".3.21..7..7...34....2........4..7.5.....5.....5.92....8.....6....7..2.19..3..1.4."}, {"row": 174, "rating": 56, "q": "..2.9.7..7....6.4....7....3.2..7.8..1....4..7..76...5..9.5.....2...1.9....8....3."}, {"row": 201, "rating": 53, "q": ".6.5...7......28......4.6..1...2.59..9.8....7..3..9...9...3..5..3.9..7.6.5.2.4..."}, {"row": 206, "rating": 62, "q": ".1...3.5.2.6.5....5.......9.....7.....4.....2.5.3...1..8..3..7.4..7.18..6..8..9.."}, {"row": 225, "rating": 62, "q": "...3.2.7..3.6..4....7.4.......5....1.1..6....8.5.2.9...2..3..4...1.....89..2..5.."}, {"row": 249, "rating": 68, "q": "..3....8.9.....1...4..9...6..25.9...4...6...2..6...7...2493...13....1.....9.25..3"}, {"row": 265, "rating": 60, "q": ".....32...5.28...127..6...89.5....4..6..5...97....9....8.....1...78....65.6.1...."}, {"row": 267, "rating": 94, "q": "..........8...25.1..61...4..2...14.54..7.......39.......52.81.7.....7....7.51.8.."}, {"row": 286, "rating": 54, "q": "....25....9.3..65....9..8.21...8..3......3..57..29............8.3...2.6.8.6..932."}, {"row": 287, "rating": 53, "q": "..2.6....8..3.......9...51......439.4.3..6..8...83.4...9..4...62.4....8...829...."}, {"row": 663, "rating": 96, "q": "..9..8.7..17.5...9......1...8.6......51.7.3..2.........9...4..5......9.3.35.9..1."}, {"row": 814, "rating": 91, "q": "..9..3.7.27..5.....3......8....1....7....6.2...65....49......41.6...2.3....4..8.."}, {"row": 1216, "rating": 118, "q": "8.......3.9....61...6..1......2..86.2...6.3.......7..265..49.....4..65..7..3....."}, {"row": 1261, "rating": 94, "q": "6...3.......6...73....126..1..3..2..7..2..35.....8......7..4...2..1..7...5......9"}, {"row": 1569, "rating": 103, "q": ".1..53.96.5.6.......6.91..5....4......51...63.7....84..2.........19...588...1...."}, {"row": 1570, "rating": 123, "q": ".2.5...3...3..9....9.7...52.3.4..5......8..23.......741....6.4...9.......4.2....5"}, {"row": 1630, "rating": 99, "q": "9.1.......7..6.1.....7....57.3.8....1.....8...8.1.6....3.61.7.9....7.3.....4...21"}, {"row": 1766, "rating": 96, "q": "....592....92...67.2........6.9..7...5.78.6....3..1....7.5..9..8......7.........4"}, {"row": 1783, "rating": 107, "q": ".1.9..46.....3........62.....1...97.9.....3.676...9.....4..6..8..6.....51..4...9."}, {"row": 1786, "rating": 123, "q": ".5.2.38....2.....3...75...2.2.9...8.7.......55.1..........6.9...9.5.2..8....4.5.."}, {"row": 1787, "rating": 141, "q": ".75.8.6...6.....39..1...5..6...4......4....8.15.6..4.....7.6.......5.8..5....2.6."}, {"row": 1788, "rating": 105, "q": "23..9...6......81..7..6...3...2.6.....3..4....6..3...93.7.....5..6.5.....5....69."}, {"row": 1854, "rating": 102, "q": "...4...7.2..........6..14.518.6.......3.4.6.1.....3.......1...8....5..9...53..1.4"}, {"row": 1857, "rating": 120, "q": ".1.4......7.......5...2...8..15.3..2....1.8..3....2...8...39..1..9...6..1...5...3"}, {"row": 1859, "rating": 109, "q": "2..9..1.8..8..3...9..1..4..1.......7..97...4.....129.........5..6.....1.4..2..7.."}, {"row": 1919, "rating": 92, "q": "8.......9...5...4...3.2.1...9.......4.62..5..2......74...15...7.8.36......1..23.."}, {"row": 1981, "rating": 152, "q": "6....5....7.3..6....4.6..1...2.5.84..5.........6.2.1........28.2....946...8.4...1"}, {"row": 1982, "rating": 98, "q": "8....57...4.8.......6.3...84....15.21....8..7....4.....9..7.8.5......2.15....2.7."}, {"row": 2025, "rating": 100, "q": "......1.2.....87.33...17.9..54.........8.....9...32..7..96.......318...91...293.."}, {"row": 2029, "rating": 97, "q": "...1.84..8...24.5...13..6.8.7..3....59.........34..5...5....3.....5...46..6..3.1."}, {"row": 2045, "rating": 96, "q": "7.5..6..361..3.7....87.9..5.......4.38.....2...6..3..8...96....8...5...9.....8.3."}, {"row": 2065, "rating": 96, "q": ".....6.....3.5.92.25.............7.93...9.45..94.......3..4.2.......9..75..1...48"}]
if SMALL:
    CANDS = CANDS[:4]
REPORT["screen"], REPORT["main"], REPORT["lyap"] = {}, {}, {}
SOLV = {}


def lyapunov(solver, puzzle, z0s, k, cap, eps_rel=1e-4, seed=1):
    """Benettin twins + QR (as v1, harness fixed): local exponents (P, cap, k) and each base trajectory's own settle."""
    m = solver.model
    dev = next(m.parameters()).device
    dt = m.inner.forward_dtype
    m.config.halt_max_steps = 10 ** 9
    Pn = len(z0s)
    N = Pn * (k + 1)
    inputs = solver.task.encode(puzzle).to(dev).repeat(N, 1)
    batch = {"inputs": inputs, "labels": torch.full_like(inputs, -100),
             "puzzle_identifiers": torch.zeros((N,), dtype=torch.int32, device=dev)}
    z = torch.stack([z0s[i] for i in range(Pn) for _ in range(k + 1)]).to(dev, dt)
    carry = _prime_eqr_carry(m, batch, z_H=z, z_L=z * 0)
    ic = carry.inner_carry
    nH, shH, shL = ic.z_H[0].numel(), ic.z_H.shape[1:], ic.z_L.shape[1:]
    base = [p * (k + 1) for p in range(Pn)]

    def state(ic):
        return torch.cat([ic.z_H.flatten(1), ic.z_L.flatten(1)], 1).double()

    def put(ic, S, Q, eps):
        for p, b in enumerate(base):
            tw = S[b][None, :] + eps[p] * Q[p].T
            ic.z_H[b + 1:b + 1 + k] = tw[:, :nH].reshape(k, *shH).to(ic.z_H.dtype)
            ic.z_L[b + 1:b + 1 + k] = tw[:, nH:].reshape(k, *shL).to(ic.z_L.dtype)

    g = torch.Generator().manual_seed(seed)
    S = state(ic)
    Q = torch.stack([torch.linalg.qr(torch.randn(S.shape[1], k, generator=g, dtype=torch.float64))[0]
                     for _ in range(Pn)]).to(dev)
    eps = eps_rel * S[base].norm(dim=1).clamp(min=1.0)
    put(ic, S, Q, eps)
    local = np.zeros((Pn, cap, k))
    grids = []
    with torch.no_grad():
        for t in range(cap):
            carry, out = m(carry, batch)
            carry.halted = torch.zeros_like(carry.halted)
            ic = carry.inner_carry
            S = state(ic)
            grids.append(np.asarray(solver.task.decode_batch(out["logits"][base]), dtype=np.int8).reshape(Pn, 81))
            newQ = []
            for p, b in enumerate(base):
                q, r = torch.linalg.qr((S[b + 1:b + 1 + k] - S[b][None, :]).T / eps[p])
                local[p, t] = torch.log(torch.abs(torch.diagonal(r))).cpu().numpy()
                newQ.append(q)
            Q = torch.stack(newQ)
            eps = eps_rel * S[base].norm(dim=1).clamp(min=1.0)
            put(ic, S, Q, eps)
    return local, suffix_settle(np.stack(grids, 1))



def run_queue(wid, items, fn):
    try:
        for it in items:
            if time.time() - T0 > GUARD_S:
                log(f"[gpu{wid}] time guard: skip", it)
                continue
            fn(wid, it)
    except Exception:
        tb = traceback.format_exc()
        log(f"[gpu{wid}] ERROR\n{tb}")
        REPORT.setdefault("errors", []).append(tb)
        save_report()


def parallel(items, fn):
    qs = {w: items[w::len(DEVS)] for w in range(len(DEVS))}
    ths = [threading.Thread(target=run_queue, args=(w, qs[w], fn)) for w in qs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


for w, d in enumerate(DEVS):
    SOLV[w] = load(torch.float32, d)


def screen(wid, c):
    T_dec, T_tau, solved = run_map(SOLV[wid], c["q"], latents(0, RES_S), CAP_S, BATCH, f"[gpu{wid}] screen r{c['row']}")
    e = escape(T_dec, CAP_S)
    REPORT["screen"][str(c["row"])] = {"rating": c["rating"], "kappa": e.get("kappa"), "r2": e.get("r2"),
                                       "median_settle": float(np.median(T_dec)),
                                       "censored_frac": float((T_dec >= CAP_S - 1).mean()), "solved": float(solved.mean())}
    save_report()
    log(f"[gpu{wid}] screen row {c['row']}: kappa {e.get('kappa')}, median settle {np.median(T_dec)}")


zc = latents(0, 4)[:8]
tr_mine, _ = run_batch(SOLV[0], HARD, zc, 12)
_r = SOLV[0].solve_batch(HARD, z_H=zc, z_L=zc * 0, max_steps=12, noise_scale=0.0, return_intermediates=True)
tr_ref = np.swapaxes(np.asarray(_r.intermediates, dtype=np.int8), 0, 1).reshape(len(zc), -1, 81)
REPORT["checks"]["stepping_equals_solve_batch"] = bool(tr_mine.shape == tr_ref.shape and np.array_equal(tr_mine, tr_ref))
save_report()
log("stepping == solve_batch:", REPORT["checks"]["stepping_equals_solve_batch"])
if not REPORT["checks"]["stepping_equals_solve_batch"]:
    raise SystemExit("own stepping loop does not reproduce solve_batch; aborting")


log("PHASE screen:", len(CANDS), "candidates")
parallel(CANDS, screen)

ok = [(v["median_settle"], int(r)) for r, v in REPORT["screen"].items()
      if MED_LO <= v["median_settle"] <= MED_HI and v["censored_frac"] <= MAX_CENS]
ok.sort()
n_q = len(ok)
if len(ok) > MAX_SEL:
    ok = [ok[i] for i in sorted(set(int(round(x)) for x in np.linspace(0, len(ok) - 1, MAX_SEL)))]
SEL = [c for c in CANDS if c["row"] in {r for _, r in ok}]
if SMALL and not SEL:   # dry-run only: exercise the main/Lyapunov path even when toy screens cannot fit kappa
    SEL = CANDS[:2]
REPORT["selected_rows"] = [c["row"] for c in SEL]
REPORT["n_qualifying"] = n_q
save_report()
log("PHASE select:", REPORT["n_qualifying"], "qualify ->", REPORT["selected_rows"])


def main_map(wid, c):
    T_dec, T_tau, solved = run_map(SOLV[wid], c["q"], latents(0, RES), CAP, BATCH, f"[gpu{wid}] main r{c['row']}")
    np.savez_compressed(f"{OUT}/main_r{c['row']}.npz", T_dec=T_dec, T_tau=T_tau, solved=solved)
    e = escape(T_dec, CAP)
    lab = lab_stats(T_dec.reshape(RES, RES), CAP - 1, RES)
    REPORT["main"][str(c["row"])] = {"rating": c["rating"], "escape": e, "lab_dec": lab,
                                     "censored_frac": float((T_dec >= CAP - 1).mean()), "solved": float(solved.mean())}
    # Lyapunov on the 8 slowest points of this map
    slow = np.argsort(-T_dec, kind="stable")[:N_LY]
    zall = latents(0, RES)
    loc, st = lyapunov(SOLV[wid], c["q"], [zall[j] for j in slow], K_LY, CAP_LY)
    has = [int(st[p]) > 3 for p in range(len(slow))]
    l1 = [float(loc[p, 2:int(st[p]), 0].mean()) for p in range(len(slow)) if has[p]]
    REPORT["lyap"][str(c["row"])] = {"settle": st.tolist(), "lambda1_points": l1,
                                     "lambda1": float(np.median(l1)) if l1 else None,
                                     "n_excluded": int(len(slow) - sum(has))}
    save_report()
    log(f"[gpu{wid}] main row {c['row']}: kappa {e.get('kappa')}, H {lab.get('H')}, alpha {lab.get('alpha')}, "
        f"lambda1 {REPORT['lyap'][str(c['row'])]['lambda1']}")


log("PHASE main:", len(SEL), "puzzles")
parallel(SEL, main_map)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


rows, nan_rows = [], []
for c in SEL:
    k = str(c["row"])
    m, ly = REPORT["main"].get(k), REPORT["lyap"].get(k)
    if not m or not ly:
        continue
    kap, lam, Hh, al = m["escape"].get("kappa"), ly["lambda1"], m["lab_dec"].get("H"), m["lab_dec"].get("alpha")
    if kap is None or lam is None:
        continue
    pred = kap / lam
    if Hh is not None and np.isfinite(Hh):
        rows.append({"row": c["row"], "kappa": kap, "lambda1": lam, "pred": pred, "H": Hh,
                     "alpha": al if al is not None and np.isfinite(al) else None})
    else:
        nan_rows.append({"row": c["row"], "kappa": kap, "lambda1": lam, "pred": pred})
res = {"test_set": rows, "H_nan": nan_rows, "n": len(rows)}
if len(rows) >= (5 if not SMALL else 2):
    H_ = np.array([r["H"] for r in rows]); P_ = np.array([r["pred"] for r in rows]); K_ = np.array([r["kappa"] for r in rows])
    res["B1_spearman_H_pred"] = spearman(H_, P_)
    res["B2_median_abs_diff"] = float(np.median(np.abs(H_ - P_)))
    res["B1_pass"] = bool(res["B1_spearman_H_pred"] >= 0.7)
    res["B2_pass"] = bool(res["B2_median_abs_diff"] <= 0.10)
    res["PASS"] = bool(res["B1_pass"] and res["B2_pass"])
    res["spearman_H_kappa_alone"] = spearman(H_, K_)
    A_ = [(r["alpha"], r["pred"]) for r in rows if r["alpha"] is not None]
    if len(A_) >= 3:
        res["alpha_spearman_pred"] = spearman(np.array([a for a, _ in A_]), np.array([p for _, p in A_]))
        res["alpha_median_abs_diff"] = float(np.median([abs(a - p) for a, p in A_]))
else:
    res["verdict"] = "UNDETERMINED (B0: fewer than 5 puzzles with finite H, kappa, lambda1)"
REPORT["test"] = res
REPORT["total_minutes"] = round((time.time() - T0) / 60, 1)
save_report()
log("TEST", json.dumps({k: v for k, v in res.items() if k not in ("test_set", "H_nan")}, default=float))
log("ALL DONE", REPORT["total_minutes"], "min")
