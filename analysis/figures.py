"""Generate figures/ (light + dark PNG for the README, PDF for the note). Run from the repository root:
python analysis/figures.py"""
import json

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import N_PAIRS, RES, ROOT, f_pred, kappa_valid, load_eps_runs, load_maps  # noqa: E402

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", ref="#8a8984",
                  s1="#2a78d6", s2="#eb6834"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", grid="#2b2b29", ref="#8a8984",
                 s1="#3987e5", s2="#d95926"),
}
EPS = [1e-1, 1e-3, 1e-5, 1e-8, 1e-12]
rng = np.random.default_rng(0)
LABEL_OFF = {102: (-24, 4), 814: (-22, -12), 225: (6, -2), 1982: (6, 2), 265: (6, -12)}


def style(ax, th):
    ax.set_facecolor(th["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["ink2"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=th["ink2"], labelsize=8, width=0.8)
    ax.grid(True, which="major", color=th["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["ink"]); ax.yaxis.label.set_color(th["ink"])
    ax.title.set_color(th["ink"])


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0), c + h


def save(fig, name, th_name):
    fig.savefig(OUT / f"{name}_{th_name}.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if th_name == "light":
        fig.savefig(OUT / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


jobs = load_eps_runs()
maps = load_maps()
rows = sorted({j["row"] for j in jobs}, key=lambda r: next(j["puzzle"]["kappa"] / j["puzzle"]["lambda1"]
                                                         for j in jobs if j["row"] == r))
slopes = {}
for run in ("eps_pairs_v5", "eps_pairs_v6"):
    for row, s in json.load(open(RES / run / "report.json"))["result"]["E2_slopes"].items():
        slopes[int(row)] = dict(s, run=run)


def slope_boot(js):
    """Bootstrap the registered slope estimator (levels with 0.03 < f < 0.97) over pairs."""
    diff = [np.array(j["Ta"]) != np.array(j["Tb"]) for j in js]
    out = []
    for _ in range(1000):
        f = [d[rng.integers(0, len(d), len(d))].mean() for d in diff]
        ok = [i for i, x in enumerate(f) if 0.03 < x < 0.97]
        if len(ok) >= 2:
            out.append(np.polyfit(np.log([js[i]["eps"] for i in ok]), np.log([f[i] for i in ok]), 1)[0])
    return np.percentile(out, [2.5, 97.5])


boot = {r: slope_boot(sorted([j for j in jobs if j["row"] == r], key=lambda j: -j["eps"])) for r in rows}

for th_name, th in THEMES.items():
    # ---- Figure 1: f(eps) per puzzle -------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True, facecolor=th["bg"])
    for ax, r in zip(axes.flat, rows):
        js = sorted([j for j in jobs if j["row"] == r], key=lambda j: -j["eps"])
        p = js[0]["puzzle"]; kl = p["kappa"] / p["lambda1"]
        e = np.array([j["eps"] for j in js]); k = np.array([j["k_differ"] for j in js])
        fp = np.array([f_pred(j) for j in js])
        ax.plot(e, np.maximum(fp, 1e-3), color=th["s2"], lw=2, zorder=2, label="predicted: mean S(horizon)")
        pos = k > 0
        lo_hi = np.array([wilson(x, N_PAIRS) for x in k])
        ax.errorbar(e[pos], k[pos] / N_PAIRS, yerr=[k[pos] / N_PAIRS - lo_hi[pos, 0], lo_hi[pos, 1] - k[pos] / N_PAIRS],
                    fmt="o", ms=5, color=th["s1"], ecolor=th["s1"], elinewidth=1, capsize=0,
                    mec=th["bg"], mew=1, zorder=3, label="observed (48 pairs, 95% CI)")
        ax.plot(e[~pos], np.full((~pos).sum(), 0.5 / N_PAIRS), "v", ms=5, mfc="none", mec=th["s1"], mew=1.2,
                zorder=3, label="observed 0/48")
        s = slopes[r]["slope"]
        ok = (k / N_PAIRS > 0.03) & (k / N_PAIRS < 0.97)
        if ok.sum() >= 2:
            x0, y0 = np.exp(np.log(e[ok]).mean()), np.exp(np.log(k[ok] / N_PAIRS).mean())
            xx = np.array([e[ok].min(), e[ok].max()])
            ax.plot(xx, y0 * (xx / x0) ** kl, ls="--", lw=1, color=th["ref"], zorder=1,
                    label=r"slope $\kappa/\lambda_1$ (theory)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.007, 1.3); ax.set_xlim(3e-14, 0.5)
        ax.set_title(f"puzzle {r}   " + r"$\kappa/\lambda_1$" + f" = {kl:.3f}", fontsize=9, loc="left")
        style(ax, th)
    for ax in axes[-1]:
        ax.set_xlabel(r"initial separation $\varepsilon$ (relative)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of pairs settling\nat different loops", fontsize=9)
    h, l = axes.flat[0].get_legend_handles_labels()
    order = [l.index(x) for x in ("observed (48 pairs, 95% CI)", "observed 0/48", "predicted: mean S(horizon)",
                                  r"slope $\kappa/\lambda_1$ (theory)")]
    h, l = [h[i] for i in order], [l[i] for i in order]
    leg = fig.legend(h, l, loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for t in leg.get_texts():
        t.set_color(th["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig1_eps_scaling", th_name)

    # ---- Figure 2: measured slope vs kappa/lambda1 ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=th["bg"])
    xs = np.array([0.04, 0.9]); ax.plot(xs, xs, color=th["ref"], lw=1, ls="--", zorder=1,
                                        label=r"slope $=\kappa/\lambda_1$")
    for run, col, mk, lab in (("eps_pairs_v5", th["s1"], "o", "v5 (5 puzzles)"), ("eps_pairs_v6", th["s2"], "s", "v6 (4 new puzzles)")):
        rr = [r for r in rows if slopes[r]["run"] == run]
        x = np.array([slopes[r]["kappa_over_lambda1"] for r in rr]); y = np.array([slopes[r]["slope"] for r in rr])
        lo = np.array([boot[r][0] for r in rr]); hi = np.array([boot[r][1] for r in rr])
        ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt=mk, ms=7, color=col, ecolor=col,
                    elinewidth=1, mec=th["bg"], mew=1.2, zorder=3, label=lab)
        for r, xi, yi in zip(rr, x, y):
            dx, dy = LABEL_OFF.get(r, (6, -10))
            ax.annotate(str(r), (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=th["ink2"])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.04, 0.9); ax.set_ylim(0.03, 0.9)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator([0.05, 0.1, 0.2, 0.5]))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.05", "0.1", "0.2", "0.5"]))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\kappa/\lambda_1$ (escape rate / Lyapunov exponent)", fontsize=9)
    ax.set_ylabel(r"measured slope of $f(\varepsilon)$", fontsize=9)
    style(ax, th)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(th["ink"])
    fig.tight_layout()
    save(fig, "fig2_slope_vs_ratio", th_name)

    # ---- Figure 3: exit-rate (hazard) curves --------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), facecolor=th["bg"])
    for ax, r in zip(axes, (1216, 1570, 225, 265)):
        T = maps[r]["T"]; cap = 150
        ks = np.arange(1, cap - 1); n = np.array([(T >= k).sum() for k in ks]); d = np.array([(T == k).sum() for k in ks])
        keep = n >= 20
        width = 1 if r in (225, 265) else (3 if r == 1570 else 6)
        xs_, ys_, lo_, hi_ = [], [], [], []
        kk = ks[keep]
        for a in range(kk.min(), kk.max() + 1, width):
            m = (ks >= a) & (ks < a + width) & keep
            if not m.any():
                continue
            N, D = n[m].sum(), d[m].sum()
            lo, hi = wilson(D, N)
            xs_.append(ks[m].mean()); ys_.append(-np.log(1 - D / N)); lo_.append(-np.log(1 - lo)); hi_.append(-np.log(1 - hi))
        xs_, ys_ = np.array(xs_), np.array(ys_)
        ax.errorbar(xs_, ys_, yerr=[ys_ - np.array(lo_), np.array(hi_) - ys_], fmt="o", ms=4, color=th["s1"],
                    ecolor=th["s1"], elinewidth=1, mec=th["bg"], mew=0.8, zorder=3,
                    label="exit rate per loop (95% CI)")
        kap = maps[r]["kappa"]
        ax.axhline(kap, color=th["s2"], lw=2, zorder=2, label=r"escape rate $\kappa$ (exponential fit)")
        ax.set_ylim(0, max(ys_.max(), kap) * 1.6)
        ax.set_title(f"puzzle {r}", fontsize=9, loc="left")
        ax.set_xlabel("loop", fontsize=9)
        style(ax, th)
    axes[0].set_ylabel("exit rate per loop", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.06))
    for t in leg.get_texts():
        t.set_color(th["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig3_exit_rate", th_name)

    # ---- Figure 4: lambda1 vs kappa ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.6), facecolor=th["bg"])
    pv = [p for p in maps.values() if kappa_valid(p) and p["lambda1"] is not None]
    allv = np.array([p["lambda1"] for p in maps.values() if p["lambda1"] is not None])
    ax.axhline(np.median(allv), color=th["ref"], lw=1, ls="--", zorder=1,
               label=rf"median $\lambda_1$ = {np.median(allv):.2f} (19 puzzles)")
    x = np.array([p["kappa"] for p in pv]); y = np.array([p["lambda1"] for p in pv])
    q = np.array([np.percentile(p["lambda1_points"], [25, 75]) for p in pv])
    ax.errorbar(x, y, yerr=[y - q[:, 0], q[:, 1] - y], fmt="o", ms=6, color=th["s1"], ecolor=th["s1"], elinewidth=1,
                mec=th["bg"], mew=1, zorder=3, label="puzzle (IQR over 8 trajectories)")
    ax.set_xscale("log"); ax.set_ylim(0, 3.2)
    ax.set_xlabel(r"escape rate $\kappa$ (per loop)", fontsize=9)
    ax.set_ylabel(r"Lyapunov exponent $\lambda_1$ (per loop)", fontsize=9)
    style(ax, th)
    leg = ax.legend(fontsize=8, frameon=False, loc="lower left")
    for t in leg.get_texts():
        t.set_color(th["ink"])
    fig.tight_layout()
    save(fig, "fig4_lambda_vs_kappa", th_name)
print("figures written to", OUT)
