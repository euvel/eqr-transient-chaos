"""v5: true-float64 eps-pair test of the predictability-horizon law on EqR (PREREG_v5.md).

For each (puzzle, eps): N pairs (base on the demo plane, partner at relative distance eps in the full latent),
both in the same batch. Records each pair's settle times and its loop-0 separation d0. Prediction per pair:
t_i = ln(0.1 / d0_i) / lambda1, f_pred = mean_i S(t_i), with S and lambda1 measured earlier (v3b / v4 maps).
Local design runs: EQR_LOCAL="row:eps,row:eps" EQR_N=12 python eqr_v5.py
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
                    "loopscape @ git+https://github.com/GilpinLab/loopscape@16b52aa1e9382f8cdd7af45a1ba7eb1efa755b3d"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "omegaconf", "einops"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from loopscape import get_solver  # noqa: E402
from loopscape.eqr.download import ensure_eqr_checkpoint  # noqa: E402
from loopscape.eqr.solver import _prime_eqr_carry  # noqa: E402

PUZZLES = json.loads(r"""{"225": {"q": "...3.2.7..3.6..4....7.4.......5....1.1..6....8.5.2.9...2..3..4...1.....89..2..5..", "kappa": 0.1415777822630261, "lambda1": 2.0680636928697282, "S": [1.0, 1.0, 1.0, 0.920625, 0.778125, 0.67625, 0.589375, 0.51375, 0.443125, 0.381875, 0.328125, 0.285625, 0.250625, 0.214375, 0.186875, 0.17, 0.145625, 0.125625, 0.103125, 0.094375, 0.0825, 0.075, 0.066875, 0.061875, 0.051875, 0.043125, 0.034375, 0.0275, 0.02375, 0.02, 0.013125, 0.0125, 0.010625, 0.006875, 0.005625, 0.004375, 0.004375, 0.003125, 0.003125, 0.001875, 0.001875, 0.001875, 0.001875, 0.001875, 0.00125, 0.00125, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 50}, "102": {"q": ".9..8......53...9.43...75....6...1...5.6...3.3....82..56.4...1......1...9..87..4.", "kappa": 0.3407788973846442, "lambda1": 2.141283469693499, "S": [1.0, 1.0, 0.898125, 0.301875, 0.275, 0.194375, 0.143125, 0.101875, 0.075625, 0.053125, 0.040625, 0.028125, 0.02, 0.013125, 0.009375, 0.005625, 0.005, 0.00375, 0.003125, 0.0025, 0.0025, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.000625, 0.000625, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 21}, "2045": {"q": "7.5..6..361..3.7....87.9..5.......4.38.....2...6..3..8...96....8...5...9.....8.3.", "kappa": 0.858004442533303, "lambda1": 2.4194255856235425, "S": [1.0, 0.9825, 0.64875, 0.2625, 0.144375, 0.051875, 0.02, 0.003125, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 16}, "80": {"q": ".7.1....9..4....8.3...897.......2.........24..3.7....6.1.....9.6.39.......7.51..3", "kappa": 0.2463236371714025, "lambda1": 2.1320290431943056, "S": [1.0, 1.0, 0.98875, 0.639375, 0.535, 0.39125, 0.2875, 0.229375, 0.17, 0.13375, 0.1075, 0.089375, 0.06625, 0.051875, 0.040625, 0.031875, 0.025625, 0.021875, 0.015625, 0.01125, 0.008125, 0.008125, 0.00625, 0.00625, 0.004375, 0.003125, 0.001875, 0.001875, 0.001875, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 30}, "814": {"q": "..9..3.7.27..5.....3......8....1....7....6.2...65....49......41.6...2.3....4..8..", "kappa": 0.11842285833416745, "lambda1": 1.9843939396073642, "S": [1.0, 0.73125, 0.53375, 0.515, 0.474375, 0.41375, 0.36375, 0.325625, 0.286875, 0.26, 0.23625, 0.21625, 0.195, 0.1775, 0.153125, 0.13625, 0.12625, 0.108125, 0.095, 0.083125, 0.073125, 0.06375, 0.0575, 0.050625, 0.04625, 0.039375, 0.034375, 0.02875, 0.025625, 0.02375, 0.021875, 0.02, 0.01875, 0.016875, 0.016875, 0.01625, 0.01375, 0.011875, 0.011875, 0.011875, 0.01125, 0.010625, 0.01, 0.01, 0.006875, 0.006875, 0.00625, 0.005, 0.004375, 0.0025, 0.0025, 0.001875, 0.001875, 0.001875, 0.001875, 0.00125, 0.00125, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 64}, "265": {"q": ".....32...5.28...127..6...89.5....4..6..5...97....9....8.....1...78....65.6.1....", "kappa": 0.40021555693995714, "lambda1": 2.063892985386575, "S": [1.0, 1.0, 0.7375, 0.631875, 0.54125, 0.35625, 0.256875, 0.155, 0.1025, 0.063125, 0.036875, 0.023125, 0.0125, 0.00875, 0.005, 0.0025, 0.0025, 0.0025, 0.00125, 0.000625, 0.000625, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 20}, "1570": {"q": ".2.5...3...3..9....9.7...52.3.4..5......8..23.......741....6.4...9.......4.2....5", "kappa": 0.12160435757157864, "lambda1": 1.8757036447229019, "S": [1.0, 1.0, 0.903125, 0.6475, 0.610625, 0.518125, 0.458125, 0.40125, 0.356875, 0.3175, 0.281875, 0.2475, 0.21875, 0.195625, 0.174375, 0.15375, 0.139375, 0.126875, 0.118125, 0.10375, 0.090625, 0.07875, 0.063125, 0.059375, 0.0525, 0.0425, 0.038125, 0.035, 0.030625, 0.02625, 0.02375, 0.0225, 0.021875, 0.0175, 0.015, 0.0125, 0.010625, 0.009375, 0.00875, 0.006875, 0.005625, 0.005625, 0.004375, 0.004375, 0.003125, 0.0025, 0.001875, 0.001875, 0.001875, 0.001875, 0.001875, 0.001875, 0.00125, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 56}, "1982": {"q": "8....57...4.8.......6.3...84....15.21....8..7....4.....9..7.8.5......2.15....2.7.", "kappa": 0.47673718638086154, "lambda1": 2.349562236709117, "S": [1.0, 0.80375, 0.588125, 0.405625, 0.28375, 0.15875, 0.09125, 0.053125, 0.033125, 0.02, 0.009375, 0.005, 0.004375, 0.003125, 0.001875, 0.000625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "cap": 16}}""")
TEST_ROWS = [814, 225, 102, 265, 2045]
EPS = [1e-1, 1e-3, 1e-5, 1e-8, 1e-12]
N = int(os.environ.get("EQR_N", "48"))
JOBS = [(r, e) for r in TEST_ROWS for e in EPS]
if os.environ.get("EQR_LOCAL"):
    JOBS = [(int(a), float(b)) for a, b in (x.split(":") for x in os.environ["EQR_LOCAL"].split(","))]
THRESH = 0.1
REPORT = {"prereg": "PREREG_v5.md", "N": N, "eps": EPS, "thresh": THRESH, "jobs": {}, "checks": {}}


def save():
    with LOCK:
        json.dump(REPORT, open(f"{OUT}/report.json", "w"), indent=1, default=float)


# ---------------------------------------------------------------- checkpoint audit
ALLOWED = {"__builtin__ dict", "__builtin__ list", "__builtin__ long", "collections OrderedDict",
           "collections defaultdict", "typing Any", "torch._utils _rebuild_tensor_v2", "torch FloatStorage"}
import hashlib  # noqa: E402
_ck = ensure_eqr_checkpoint(task="sudoku")
_sha = hashlib.sha256(open(_ck, "rb").read()).hexdigest()
REPORT["checks"]["checkpoint_sha256"] = _sha
if _sha != "258b1f0a90003a2db0de0c8df7831683ffce00de295ed4bba15dc8e599920d26":
    REPORT["abort"] = f"EqR checkpoint changed: {_sha}"
    save()
    raise SystemExit(REPORT["abort"])
refs = set()
with zipfile.ZipFile(_ck) as z:
    for n in z.namelist():
        if n.endswith("data.pkl"):
            st = []
            for op, arg, _ in pickletools.genops(io.BytesIO(z.read(n))):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    st.append(arg)
                elif op.name == "GLOBAL":
                    refs.add(arg)
                elif op.name == "STACK_GLOBAL":
                    refs.add(f"{st[-2]} {st[-1]}")
bad = sorted(r for r in refs if r not in ALLOWED and not r.startswith("omegaconf."))
if bad:
    REPORT["abort"] = f"unexpected pickle globals {bad}"
    save()
    raise SystemExit(REPORT["abort"])
DEVS = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
if not torch.cuda.is_available():
    torch.set_num_threads(8)
REPORT["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or ["cpu"]
log("checkpoint audit ok; devices", REPORT["devices"])


# ---------------------------------------------------------------- true float64 model
def _rms_norm_keep(h, variance_epsilon):
    # upstream EqR rms_norm casts to float32 internally; keep the input precision instead
    return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + variance_epsilon)


def load64(dev):
    s = get_solver("eqr", device=dev, dtype=torch.float64, config_overrides={"forward_dtype": "float64"})
    s.model.config.noise_scale = 0.0
    s.model.inner.L_level.noise_scale = 0.0
    assert s.model.inner.forward_dtype == torch.float64
    return s


SOLV = {w: load64(d) for w, d in enumerate(DEVS)}
patched = []
for name in [n for n in list(sys.modules) if n.startswith("models.")]:
    if hasattr(sys.modules[name], "rms_norm"):
        setattr(sys.modules[name], "rms_norm", _rms_norm_keep)
        patched.append(name)
REPORT["checks"]["rms_norm_patched_in"] = patched
assert "models.eqr" in patched, patched

torch.manual_seed(0)
_q, _ = torch.linalg.qr(torch.randn(97 * 512, 2))
U, V = _q[:, 0].reshape(97, 512).double(), _q[:, 1].reshape(97, 512).double()
_g = torch.Generator().manual_seed(1)
PTS = torch.rand(N, 2, generator=_g, dtype=torch.float64) * 2 - 1
Z0 = PTS[:, 0, None, None] * U + PTS[:, 1, None, None] * V
W = torch.randn(N, 97, 512, generator=_g, dtype=torch.float64)
W = W / W.flatten(1).norm(dim=1)[:, None, None]


def suffix(tr):
    if tr.shape[1] < 2:                      # a single loop has no change to detect
        return np.zeros(tr.shape[0], dtype=int)
    ch = (tr[:, 1:] != tr[:, :-1]).any(-1)
    return np.where(ch, np.arange(1, tr.shape[1]), 0).max(1)


@torch.no_grad()
def run_pairs(s, puzzle, eps, cap):
    dev = next(s.model.parameters()).device
    m = s.model
    Zp = Z0 + eps * Z0.flatten(1).norm(dim=1)[:, None, None] * W
    z = torch.cat([Z0, Zp]).to(dev)
    B = 2 * N
    m.config.halt_max_steps = cap
    inp = s.task.encode(puzzle).to(dev).repeat(B, 1)
    batch = {"inputs": inp, "labels": torch.full_like(inp, -100),
             "puzzle_identifiers": torch.zeros((B,), dtype=torch.int32, device=dev)}
    carry = _prime_eqr_carry(m, batch, z_H=z, z_L=z * 0)
    grids, dist = [], []
    for _ in range(cap):
        carry, out = m(carry, batch)
        grids.append(np.asarray(s.task.decode_batch(out["logits"]), dtype=np.int8).reshape(B, 81))
        zh = carry.inner_carry.z_H.flatten(1)
        dist.append(((zh[N:] - zh[:N]).norm(dim=1) / zh[:N].norm(dim=1)).double().cpu().numpy())
        if bool(carry.halted.all()):
            break
    T = suffix(np.stack(grids, 1))
    return T[:N], T[N:], np.stack(dist, 1)


def S_of(S, t):
    if t <= 0:
        return 1.0
    lo = int(np.floor(t))
    if lo + 1 >= len(S):
        return S[-1]
    w = t - lo
    return (1 - w) * S[lo] + w * S[lo + 1]


def predict(p, d0):
    lam, S = p["lambda1"], p["S"]
    t = np.log(THRESH / np.maximum(d0, 1e-300)) / lam
    return float(np.mean([S_of(S, ti) for ti in t]))


# ---------------------------------------------------------------- checks (must pass)
s0 = SOLV[0]
pz0 = PUZZLES[str(TEST_ROWS[0])]["q"] if str(TEST_ROWS[0]) in PUZZLES else list(PUZZLES.values())[0]["q"]
_, _, d3 = run_pairs(s0, pz0, 1e-3, 1)
_, _, d6 = run_pairs(s0, pz0, 1e-6, 1)
ratio = float(np.median(d3[:, 0] / d6[:, 0]))
REPORT["checks"]["loop0_separation_ratio_1e-3_over_1e-6"] = ratio
REPORT["checks"]["true_fp64_scaling_ok"] = bool(abs(ratio / 1000 - 1) < 0.01)
save()
log("true-fp64 check: d(1e-3)/d(1e-6) at loop 0 =", round(ratio, 2), "(must be ~1000)")
if not REPORT["checks"]["true_fp64_scaling_ok"]:
    raise SystemExit("loop-0 separation does not scale with eps: not true float64")


@torch.no_grad()
def own_traces(s, puzzle, z, cap):
    dev = next(s.model.parameters()).device
    m = s.model
    m.config.halt_max_steps = cap
    inp = s.task.encode(puzzle).to(dev).repeat(len(z), 1)
    batch = {"inputs": inp, "labels": torch.full_like(inp, -100),
             "puzzle_identifiers": torch.zeros((len(z),), dtype=torch.int32, device=dev)}
    carry = _prime_eqr_carry(m, batch, z_H=z.to(dev), z_L=z.to(dev) * 0)
    gr = []
    for _ in range(cap):
        carry, out = m(carry, batch)
        gr.append(np.asarray(s.task.decode_batch(out["logits"]), dtype=np.int8).reshape(len(z), 81))
        if bool(carry.halted.all()):
            break
    return np.stack(gr, 1)


_nc = min(4, N)
_r = s0.solve_batch(pz0, z_H=Z0[:_nc], z_L=Z0[:_nc] * 0, max_steps=6, noise_scale=0.0, return_intermediates=True)
_ref = np.swapaxes(np.asarray(_r.intermediates, dtype=np.int8), 0, 1).reshape(_nc, -1, 81)
_own = own_traces(s0, pz0, Z0[:_nc], 6)
REPORT["checks"]["stepping_equals_solve_batch_fp64"] = bool(_own.shape == _ref.shape and np.array_equal(_own, _ref))
save()
log("stepping == solve_batch (fp64, patched):", REPORT["checks"]["stepping_equals_solve_batch_fp64"])
if not REPORT["checks"]["stepping_equals_solve_batch_fp64"]:
    raise SystemExit("own stepping loop does not reproduce solve_batch; aborting")


def job(wid, r, e):
    p = PUZZLES[str(r)]
    t = time.time()
    Ta, Tb, D = run_pairs(SOLV[wid], p["q"], e, p["cap"])
    d0 = D[:, 0]
    f_obs = float((Ta != Tb).mean())
    rec = {"row": r, "eps": e, "cap": p["cap"], "f_obs": f_obs, "k_differ": int((Ta != Tb).sum()),
           "f_pred_perpair": predict(p, d0),
           "f_pred_median_c": predict(p, np.full(N, np.median(d0))),
           "f_pred_c0039": predict(p, np.full(N, 0.039 * e)),
           "c_median": float(np.median(d0 / e)), "Ta": Ta.tolist(), "Tb": Tb.tolist(), "d0": d0.tolist(),
           "d_median_by_loop": np.median(D, 0).tolist(), "censored_pairs": int(((Ta >= p["cap"] - 1) & (Tb >= p["cap"] - 1)).sum()),
           "seconds": round(time.time() - t)}
    REPORT["jobs"][f"{r}:{e:g}"] = rec
    save()
    log(f"[gpu{wid}] row {r} eps {e:g}: f_obs {f_obs:.3f} ({rec['k_differ']}/{N})  f_pred {rec['f_pred_perpair']:.3f}  "
        f"c_med {rec['c_median']:.4f}  [{rec['seconds']}s]")


def worker(wid, items):
    try:
        for r, e in items:
            if time.time() - T0 > GUARD_S:
                log(f"[gpu{wid}] time guard: skip {r}:{e:g}")
                continue
            job(wid, r, e)
    except Exception:
        tb = traceback.format_exc()
        log(f"[gpu{wid}] ERROR\n{tb}")
        REPORT.setdefault("errors", []).append(tb)
        save()


# balance by cost (cap): longest jobs first, greedy onto the less-loaded worker
order = sorted(JOBS, key=lambda j: -PUZZLES[str(j[0])]["cap"])
queues, load = {w: [] for w in SOLV}, {w: 0 for w in SOLV}
for jb in order:
    w = min(load, key=load.get)
    queues[w].append(jb)
    load[w] += PUZZLES[str(jb[0])]["cap"]
ths = [threading.Thread(target=worker, args=(w, q)) for w, q in queues.items()]
for th in ths:
    th.start()
for th in ths:
    th.join()


# ---------------------------------------------------------------- bars (PREREG_v5.md)
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


pts = [j for j in REPORT["jobs"].values() if 0.05 <= j["f_pred_perpair"] <= 0.95]
res = {"n_points": len(pts), "n_jobs_done": len(REPORT["jobs"])}
done_per_puzzle = {r: sum(1 for j in REPORT["jobs"].values() if j["row"] == r) for r in TEST_ROWS}
res["E0_puzzles_complete"] = int(sum(1 for r in TEST_ROWS if done_per_puzzle[r] == len(EPS)))
res["E0_pass"] = bool(res["E0_puzzles_complete"] >= 4)
if len(pts) >= 3:
    fp = np.array([j["f_pred_perpair"] for j in pts])
    fo = np.array([(j["k_differ"] + 0.5) / (N + 1) for j in pts])
    lr = np.abs(np.log(fo / fp))
    res["E1a_spearman"] = spearman(fp, fo)
    res["E1b_median_abs_log_ratio"] = float(np.median(lr))
    res["E1c_frac_within_factor2"] = float(np.mean(lr <= np.log(2)))
    res["E1_pass"] = bool(res["E1a_spearman"] >= 0.8 and res["E1b_median_abs_log_ratio"] <= np.log(2)
                          and res["E1c_frac_within_factor2"] >= 0.7)
slopes = {}
for r in TEST_ROWS:
    js = sorted([j for j in REPORT["jobs"].values() if j["row"] == r and 0.03 < j["f_obs"] < 0.97], key=lambda j: j["eps"])
    if len(js) >= 2:
        x, y = np.log([j["eps"] for j in js]), np.log([j["f_obs"] for j in js])
        slopes[r] = {"slope": float(np.polyfit(x, y, 1)[0]),
                     "kappa_over_lambda1": PUZZLES[str(r)]["kappa"] / PUZZLES[str(r)]["lambda1"], "n_levels": len(js)}
res["E2_slopes"] = slopes
if len(slopes) >= 3:
    res["E2_spearman_slope_vs_kl"] = spearman(np.array([v["slope"] for v in slopes.values()]),
                                              np.array([v["kappa_over_lambda1"] for v in slopes.values()]))
REPORT["result"] = res
REPORT["total_minutes"] = round((time.time() - T0) / 60, 1)
save()
log("RESULT", json.dumps(res, default=float))
log("ALL DONE", REPORT["total_minutes"], "min")
