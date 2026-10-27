# Raw outputs

Everything here was written by the kernels in `../experiments/` (Kaggle, T4 ×2) and is used unchanged.

| Folder | Run | Contents |
|---|---|---|
| `maps_v3b/`, `maps_v4/` | settle-time maps + Lyapunov exponents (float32) | `main_r<row>.npz`: `T_dec` (settle time per pixel, 40×40 map flattened, cap 150; `T = cap - 1` = still changing), `T_tau` (alternative settle time: first loop with relative step < 0.05), `solved`. `report.json` → `main[row].escape` (κ fit: `kappa`, `r2`, `range`, `kappa_ci95`), `main[row].lab_dec` (loopscape roughness estimators), `lyap[row]` (`lambda1` = median over `lambda1_points`, one per slow trajectory). `log.txt` = kernel log. |
| `maps_v2/` | long-cap maps from an earlier run (float32, cap 120) | `HARD_p1.npz`, `P3_p0.npz`, `P3_p1.npz`; used only in `analysis/survival_shape.py`. |
| `eps_pairs_v5/`, `eps_pairs_v6/` | ε-pairs (true float64) | `report.json` → `jobs["<row>:<eps>"]`: `Ta`, `Tb` (settle times of the 48 base / partner points), `d0` (relative separation after loop 0), `k_differ`, `f_pred_*`, `d_median_by_loop`; `checks` (in-kernel precision checks); `result` (registered bars, as computed in-kernel). `puzzles_v*.json` = the puzzle inputs (question string, κ, λ₁, survival curve S, cap). |

Puzzle rows index the Sudoku-Extreme test set (`sapientinc/sudoku-extreme`, `test.csv`) in file order, as listed in
`../experiments/01_settle_maps_and_lyapunov/candidates_v3.json`.
