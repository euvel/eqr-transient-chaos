# Experiments

Each script is a self-contained Kaggle kernel (GPU, internet on). It installs loopscape, downloads the EqR checkpoint,
checks its SHA-256 and audits the pickle globals before loading, then writes `out/report.json` + `out/log.txt`.
Locally: `EQR_SMALL=1` (maps) or `EQR_LOCAL="row:eps" EQR_N=2` (ε-pairs) run tiny CPU versions.

| Folder | Script | Pre-registration | Output in `results/` |
|---|---|---|---|
| `00_long_maps_v2/` | `eqr_v2.py` (exploratory robustness run; only its HARD/P3 long-cap maps are used) | — | `maps_v2/` |
| `01_settle_maps_and_lyapunov/` | `eqr_v3b.py`, `eqr_v4_p0.py` (+ `candidates_v3.json`, `pool_v4.json`) | `PREREG_v3.md`, `PREREG_v4_confirmation.md` | `maps_v3b/`, `maps_v4/` |
| `02_eps_pairs_v5/` | `eqr_v5.py` | `PREREG_v5.md` | `eps_pairs_v5/` |
| `03_eps_pairs_v6/` | `eqr_v6.py`, `make_puzzles_v6.py` | `PREREG_v6.md` | `eps_pairs_v6/` |

Notes:
- `eqr_v5.py` / `eqr_v6.py` pin loopscape to commit `16b52aa`. The earlier scripts install loopscape's default branch,
  which was the same commit at run time (no later commits exist as of 27 Oct 2026).
- True float64 in the ε-pair kernels = `config_overrides={"forward_dtype": "float64"}` plus a precision-preserving
  `rms_norm`; both are verified in-kernel before any job (see `../repro/`).
- The kernels refer to their pre-registration files by their original working paths; the files themselves are in
  `../preregistration/`.
