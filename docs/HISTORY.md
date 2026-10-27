# Project history (25–27 October 2026)

This log is chronological and includes the dead ends. Runs marked † produced data that is **not** in this repository:
they were exploratory, or affected by the precision problems described below. Their conclusions are used here only
qualitatively.

## Exploration
- **v1†:** Reproduced loopscape's EqR settle-time maps on Kaggle. The first uncertainty-exponent estimate used a
  simple two-outcome formula that is wrong for a many-valued settle-time field, so we re-scored with loopscape's own
  estimators. The run contained a "float64" comparison that later turned out to be float32 (see *Precision*).
- **v2†:** Robustness on five puzzles: repeat runs, batch-size changes, float32 sub-grids, two slices, and first
  $\lambda_1$ estimates.
  - The settle-time **statistics** were robust.
  - Individual **pixels** of slow puzzles were not. Changing the batch composition changed 93% of pixels on the
    hardest puzzle.
  - The idea that rounding, amplified by chaos, decides those pixels led to the predictability-horizon hypothesis.
  - The long-cap maps of puzzles HARD and P3 from this run (float32, unaffected by the precision issue) are in
    `results/maps_v2/` and are used only in the survival-shape check.
- **FPRM maze (loopscape's `fractal.py` pipeline)†:** bit-reproducible across 16/16 repeat cells. Its total
  log-stretch over a transient (2.7–4.7) stays below float32's 16.6, so rounding never reaches O(1): its slowness is
  near-marginal drift, not chaos. At step size $\eta = 1$ the maze recurrence enters a period-2 cycle, so all FPRM work
  used $\eta = 0.5$.

## Pre-registered test of the lab's roughness estimator (v3 → v3b, v4)
- **v3** was cancelled after 20 minutes of screening, before any main result. The selection rule favoured slow
  puzzles, on which the estimator is undefined (Amendment 1).
- **v3b** and **v4 (part 0)** measured $\kappa$ and $\lambda_1$ on 19 puzzles: this repository's `results/maps_*`.
  Their registered test ($H$ against $\kappa/\lambda_1$) was **undetermined**: only 3 puzzles had a finite $H$. Part 1
  of v4 was held and never run.
- What these runs did establish: $\lambda_1 \approx 2.07$ per loop for every puzzle, and $\kappa$ varying 20-fold.

## Predictability horizon (local, then v5, v6)
- **Hypothesis:** two starting points share a settle time unless thinking outlasts the horizon
  $t = \ln(0.1/d_0)/\lambda_1$. This gives $f(\varepsilon) \approx \langle S(t)\rangle \propto
  \varepsilon^{\kappa/\lambda_1}$.
- **Laptop tests** found the second precision problem. After the fix, a small pre-registered test on one puzzle passed.
  A first version of the horizon formula (a constant offset) failed and was replaced by the measured
  first-loop separation $d_0$.
- **v5** (5 puzzles, true float64): the exponent matched. The amplitude **failed** its bar (61% within 2×, bar 70%).
- **Post-hoc analysis:** the law almost never wrongly predicted "same settle time" (6 of 755 pairs), but its "different"
  predictions were right only 61% of the time. A constant delay of 1.75 loops fixed this on the v5 data.
- **v6** (4 new puzzles, delay frozen at 1.75): the registered bar **passed**, but the registered contrast (no delay)
  did better, the refitted delay was 0.75 (outside the predicted band), and the sharpest point favoured no delay. **The
  delay was rejected.** The bars were too loose to separate the two versions; the contrast should have been a bar.
- **Post-hoc:** part of v5's amplitude miss came from the input survival curve. The float32 map overstated the true
  float64 survival of the run's own starting points for two puzzles.

## Checks against the literature (27 Oct)
- **Theory:** confirmed that $\alpha = \kappa/\lambda_{\max}$ is the correct high-dimensional form.
- **Lai et al.:** the paper does not measure $\kappa$ or $\lambda_1$. It expects doubly transient chaos, and a public
  referee report argues that the boundaries are slim.
- **Survival-shape and slimness checks** (`analysis/survival_shape.py`, `analysis/slim_check.py`): decision rules were
  written into each script before it was run. In both cases the answer was no for EqR (README, results 3 and 4).

## Precision
Two independent problems made "float64" EqR runs float32:
1. `EqRSolver.load` maps `torch.float64` to `"float32"`.
2. Upstream EqR's `rms_norm` casts to float32 and back.

The early float64 comparisons (v1, v2 sub-grids) were therefore float32 against float32 rounding; they were corrected
in our notes and are not used here. All $\varepsilon$-pair data in this repository come from runs that patch both and
verify true float64 in-kernel. Reported as [GilpinLab/loopscape#1](https://github.com/GilpinLab/loopscape/issues/1).
