"""v2 robustness round on EqR (GilpinLab/loopscape). Pre-registration: PREREG_v2.md (H1-H6).

Two GPU workers; each owns a queue of maps and saves every map as soon as it is done.
"""

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
REPORT = {"prereg": "PREREG_v2.md", "devices": [torch.cuda.get_device_name(i) for i in range(NDEV)]
          if torch.cuda.is_available() else ["cpu"], "torch": torch.__version__, "maps": {}, "checks": {}}

RES, CAP, CAP_EASY, RES_SUB, CAP_SUB, CAP_SUB_EASY, BATCH = 40, 120, 40, 20, 60, 20, 256
N_LY, K_LY, CAP_LY = 8, 4, 100
if SMALL:
    RES, CAP, CAP_EASY, RES_SUB, CAP_SUB, CAP_SUB_EASY, BATCH = 4, 8, 6, 2, 6, 4, 16
    N_LY, K_LY, CAP_LY = 2, 2, 6


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
        if b - a < 5 or S[b] <= 0 or S[a] / S[b] < np.e:
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


# ---------------------------------------------------------------- work items
SUB = np.array([i * RES + j for i in range(0, RES, 2) for j in range(0, RES, 2)])[: RES_SUB * RES_SUB]
H_MAPS = [("P1", 0), ("P1", 1), ("P2", 0), ("P2", 1), ("P3", 0), ("P3", 1), ("HARD", 1)]
QUEUES = {
    0: [("P1", 0, CAP, BATCH, torch.float32, None, "P1_p0"),
        ("P1", 0, CAP, BATCH, torch.float32, None, "P1_p0_repeat"),
        ("P1", 0, CAP, 100 if not SMALL else 7, torch.float32, None, "P1_p0_b100"),
        ("P2", 0, CAP, BATCH, torch.float32, None, "P2_p0"),
        ("P3", 0, CAP, BATCH, torch.float32, None, "P3_p0"),
        ("EASY", 0, CAP_EASY, BATCH, torch.float32, None, "EASY_p0"),
        ("EASY", 0, CAP_EASY, BATCH, torch.float32, None, "EASY_p0_repeat"),
        ("EASY", 0, CAP_EASY, 100 if not SMALL else 7, torch.float32, None, "EASY_p0_b100"),
        "LYAP"],
    1: [("P1", 1, CAP, BATCH, torch.float32, None, "P1_p1"),
        ("P2", 1, CAP, BATCH, torch.float32, None, "P2_p1"),
        ("P3", 1, CAP, BATCH, torch.float32, None, "P3_p1"),
        ("HARD", 1, CAP, BATCH, torch.float32, None, "HARD_p1"),
        ("P1", 0, CAP_SUB, 32, torch.float32, SUB, "P1_p0_fp32sub"),
        ("EASY", 0, CAP_SUB_EASY, 32, torch.float32, SUB, "EASY_p0_fp32sub"),
        ("P1", 0, CAP_SUB, 32, torch.float64, SUB, "P1_p0_fp64sub"),
        ("EASY", 0, CAP_SUB_EASY, 32, torch.float64, SUB, "EASY_p0_fp64sub")],
}
if NDEV == 1:   # CPU dry run or single GPU: one worker does everything
    QUEUES = {0: [q for q in QUEUES[0] if q != "LYAP"] + QUEUES[1] + ["LYAP"]}
RAW = {}


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


def worker(wid, dev):
    try:
        solvers = {}
        for item in QUEUES[wid]:
            if time.time() - T0 > GUARD_S:
                log(f"[gpu{wid}] time guard: skipping", item if isinstance(item, str) else item[-1])
                REPORT["maps"][item if isinstance(item, str) else item[-1]] = {"skipped": "time guard"}
                continue
            if item == "LYAP":
                s = solvers[torch.float32]
                rows = {}
                for pz in ("P1", "P2", "P3"):
                    if f"{pz}_p0" not in RAW:
                        continue
                    T_dec = RAW[f"{pz}_p0"][0]
                    slow = np.argsort(-T_dec, kind="stable")[:N_LY]
                    zall = latents(0, RES)
                    loc, st = lyapunov(s, P[pz], [zall[j] for j in slow], K_LY, CAP_LY)
                    np.save(f"{OUT}/LYAP_{pz}_local.npy", loc)
                    # transient window = loops 2..settle-1 (2-loop burn-in); points with no such window are
                    # excluded and counted, never averaged over post-settle loops
                    has = [int(st[p]) > 3 for p in range(len(slow))]
                    l1 = [float(loc[p, 2:int(st[p]), 0].mean()) for p in range(len(slow)) if has[p]]
                    spec = [loc[p, 2:int(st[p])].mean(0).tolist() for p in range(len(slow)) if has[p]]
                    pos = float(np.mean(np.array(l1) > 0)) if l1 else None
                    rows[pz] = {"idx": slow.tolist(), "settle": st.tolist(), "lambda1": l1,
                                "n_without_transient_window": int(len(slow) - sum(has)),
                                "spectrum_median": np.median(np.array(spec), 0).tolist() if spec else None,
                                "lambda1_pos_frac": pos,
                                "H2_pass": None if pos is None else bool(pos >= 0.9)}
                    log(f"[gpu{wid}] LYAP {pz}: lambda1 median {np.median(l1) if l1 else None}, pos frac {pos}, "
                        f"excluded {rows[pz]['n_without_transient_window']}")
                REPORT["maps"]["LYAP"] = rows
                save_report()
                continue
            pz, seed, cap, batch, dtype, sub, name = item
            if dtype not in solvers:
                solvers[dtype] = load(dtype, dev)
            z = latents(seed, RES, sub)
            if dtype == torch.float64:
                z = z.double()
            t = time.time()
            T_dec, T_tau, solved = run_map(solvers[dtype], P[pz], z, cap, batch, f"[gpu{wid}] {name}")
            RAW[name] = (T_dec, T_tau, solved)
            np.savez_compressed(f"{OUT}/{name}.npz", T_dec=T_dec, T_tau=T_tau, solved=solved,
                                sub=sub if sub is not None else -1)
            res = RES if sub is None else RES_SUB
            summ = summarize(T_dec, T_tau, solved, cap, res)
            summ.update({"puzzle": pz, "plane": seed, "batch": batch, "dtype": str(dtype), "device": dev,
                         "seconds": round(time.time() - t)})
            REPORT["maps"][name] = summ
            save_report()
            e = summ["escape_dec"]
            log(f"[gpu{wid}] DONE {name}: kappa {e.get('kappa')}, R2 {e.get('r2')}, censored@24 {summ['dec_censored_at24']}, "
                f"lab H(dec) {summ['lab_dec_full'].get('H')}, solved {summ['solved_frac']:.3f}")
    except Exception:
        tb = traceback.format_exc()
        log(f"[gpu{wid}] ERROR\n{tb}")
        REPORT.setdefault("errors", []).append(tb)
        save_report()


# ---------------------------------------------------------------- equivalence check (must pass)
s0 = load(torch.float32, DEVS[0])
zc = latents(0, 4)[:8]
tr_mine, _ = run_batch(s0, HARD, zc, 12)
r = s0.solve_batch(HARD, z_H=zc, z_L=zc * 0, max_steps=12, noise_scale=0.0, return_intermediates=True)
tr_ref = np.swapaxes(np.asarray(r.intermediates, dtype=np.int8), 0, 1).reshape(len(zc), -1, 81)
eq = bool(tr_mine.shape == tr_ref.shape and np.array_equal(tr_mine, tr_ref))
REPORT["checks"]["stepping_equals_solve_batch"] = eq
save_report()
log("stepping == solve_batch:", eq)
if not eq:
    raise SystemExit("own stepping loop does not reproduce solve_batch; aborting")
del s0

threads = [threading.Thread(target=worker, args=(w, DEVS[w])) for w in QUEUES]
for t in threads:
    t.start()
for t in threads:
    t.join()


# ---------------------------------------------------------------- hypotheses
M = REPORT["maps"]
H = {}
fits = [M.get(f"{p}_p{s}" if p != "HARD" else "HARD_p1", {}).get("escape_dec", {}) for p, s in H_MAPS]
ok = [bool(f.get("r2") is not None and f["r2"] >= 0.97) for f in fits]
H["H1"] = {"maps_pass": int(sum(ok)), "of": len(ok), "pass": bool(sum(ok) >= 6 if not SMALL else True)}
ly = M.get("LYAP", {})
H["H2"] = {pz: ly[pz]["H2_pass"] for pz in ly} | {"pass": bool(ly) and all(ly[pz]["H2_pass"] is True for pz in ly)}


def agree(a, b, sub=None):
    if a not in RAW or b not in RAW:
        return None
    A, B = RAW[a], RAW[b]
    if sub is not None:
        A = tuple(x[sub] for x in A)
    n = min(len(A[0]), len(B[0]))
    return {"T_dec": float((A[0][:n] == B[0][:n]).mean()), "T_tau": float((A[1][:n] == B[1][:n]).mean())}


H["H3"] = {"P1": agree("P1_p0", "P1_p0_repeat"), "EASY": agree("EASY_p0", "EASY_p0_repeat")}
H["H3"]["pass"] = bool(all(v and v["T_dec"] == 1.0 and v["T_tau"] == 1.0 for v in (H["H3"]["P1"], H["H3"]["EASY"])))
H["H4"] = {"P1_batch": agree("P1_p0", "P1_p0_b100"), "EASY_batch": agree("EASY_p0", "EASY_p0_b100"),
           # fp32 twin of the sub-grid: same points, cap and batch, so precision is the only difference
           "P1_fp64": agree("P1_p0_fp32sub", "P1_p0_fp64sub"),
           "EASY_fp64": agree("EASY_p0_fp32sub", "EASY_p0_fp64sub")}
try:
    H["H4"]["pass"] = bool(H["H4"]["P1_batch"]["T_dec"] < 0.5 and H["H4"]["P1_fp64"]["T_dec"] < 0.5
                           and H["H4"]["EASY_batch"]["T_dec"] >= 0.95 and H["H4"]["EASY_fp64"]["T_dec"] >= 0.95)
except Exception as e:
    H["H4"]["pass"] = None
    H["H4"]["error"] = repr(e)
try:
    a, b = M["P1_p0"], M["P1_p0_b100"]
    ka, kb = a["escape_dec"]["kappa"], b["escape_dec"]["kappa"]
    ha, hb = a["lab_dec_full"].get("H"), b["lab_dec_full"].get("H")
    h_ok = None if (ha is None or hb is None or not np.isfinite(ha) or not np.isfinite(hb)) else bool(abs(ha - hb) <= 0.05)
    H["H5"] = {"kappa": [ka, kb], "kappa_ok": bool(abs(ka - kb) / ka <= 0.10),
               "cens24": [a["dec_censored_at24"], b["dec_censored_at24"]],
               "cens24_ok": bool(abs(a["dec_censored_at24"] - b["dec_censored_at24"]) <= 0.03),
               "lab_H": [ha, hb], "lab_H_ok": h_ok}
    H["H5"]["pass"] = bool(H["H5"]["kappa_ok"] and H["H5"]["cens24_ok"] and h_ok is not False)
except Exception as e:
    H["H5"] = {"pass": None, "error": repr(e)}
REPORT["hypotheses"] = H
REPORT["total_minutes"] = round((time.time() - T0) / 60, 1)
save_report()
log("HYPOTHESES", json.dumps(H, default=float)[:2000])
log("ALL DONE", REPORT["total_minutes"], "min")
