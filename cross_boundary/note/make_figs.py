"""Figures for the hidden-roughness note, read from the saved results in ../cross_boundary/."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CB = Path(__file__).resolve().parents[1] / "toys"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "font.family": "sans-serif"})

# ---- Fig 1: the two precision laws (measured slope vs predicted exponent)
a1 = json.load(open(CB / "a1_precision_law.json"))
a1b = json.load(open(CB / "a1b_contamination_law.json"))
x1 = [v["h"] for v in a1.values()]
y1 = [v["slope_logfloor_vs_logeps"] for v in a1.values()]
x2 = [v["h_over_1mh"] for v in a1b.values()]
y2 = [v["slope"] for v in a1b.values()]
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.plot([0, 1.85], [0, 1.85], color=MUTED, lw=1, ls="--", zorder=1)
ax.text(1.22, 1.02, "measured = predicted", color=MUTED, rotation=36, fontsize=7)
ax.scatter(x1, y1, s=36, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
           label=r"rounding law, $p=h$")
ax.scatter(x2, y2, s=36, color=ORANGE, edgecolor="white", linewidth=1.5, marker="s", zorder=3,
           label=r"direction-error law, $p=h/(1-h)$")
ax.set_xlabel("predicted exponent $p$")
ax.set_ylabel("measured exponent")
ax.set_xlim(0, 1.85)
ax.set_ylim(0, 1.85)
ax.grid(color=GRID, lw=0.6)
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_precision_laws.{ext}", dpi=200)

# ---- Fig 2: kicked float64 orbits vs genuine 80-digit orbits under feedback
g = json.load(open(CB / "s0g_feedback.json"))
h = json.load(open(CB / "s0h_true_orbit.json"))
eps_lab = {"0": 0, "0.0": 0, "1e-3": 1, "0.001": 1, "1e-2": 2, "0.01": 2}
fig, ax = plt.subplots(figsize=(3.3, 2.4))
ax.axhline(math.log(1.5), color=MUTED, lw=1, ls="--")
ax.text(2.25, math.log(1.5) - 0.07, r"true $\lambda_x$", color=MUTED, fontsize=7, ha="right")
ax.axhline(math.log(3), color=MUTED, lw=0.6, ls=":")
ax.text(2.25, math.log(3) + 0.03, r"$\lambda_1$", color=MUTED, fontsize=7, ha="right")
gx = [eps_lab[k] - 0.08 for k in g]
gy = [v["lambda_x_hat"] for v in g.values()]
ax.scatter(gx, gy, s=36, color=ORANGE, marker="s", edgecolor="white", linewidth=1.5, zorder=3,
           label="kicked float64 orbit")
hx, hy = [], []
for k, rows in h.items():
    for r in rows:
        hx.append(eps_lab[k] + 0.08)
        hy.append(r["lambda_x_hat"])
ax.scatter(hx, hy, s=30, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3, label="genuine 80-digit orbit (x3)")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["0", r"$10^{-3}$", r"$10^{-2}$"])
ax.set_xlabel(r"feedback $\epsilon$ (decision $\to$ churn)")
ax.set_ylabel(r"estimated $\lambda_x$")
ax.set_ylim(0, 1.2)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.64), fontsize=7)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_orbit_obstruction.{ext}", dpi=200)
print("ok")
