"""Figure 2 for the manuscript, drawn from the result tables.

    python make_figures.py <results_root> <out_dir>

<results_root> holds ambignq/ and pacific/, each with results/policies_by_profile.csv.
"""
import glob, sys
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1]
OUT = sys.argv[2]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL = {"staff": "#2a78d6", "agent_noticed": "#eb6834", "agent_unnoticed": "#1baf7a"}
MARK = {"staff": "o", "agent_noticed": "s", "agent_unnoticed": "D"}
LABEL = {"staff": "Staff", "agent_noticed": "Agent, failure noticed", "agent_unnoticed": "Agent, failure found later"}
POL = [("always_local", "Always local"), ("always_private", "Always private cloud"),
       ("always_frontier", "Always frontier"), ("always_ask_frontier", "Always ask, then frontier"),
       ("router_without_ask", "Router without ask"), ("router_with_ask", "Router with ask"), ("oracle", "Oracle")]
DS = [("pacific", "PACIFIC (financial reports)"), ("ambignq", "AmbigNQ (open domain)")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6})


def res(ds, f):
    return pd.read_csv((glob.glob(f"{ROOT}/{ds}/run_*/results/{f}.csv") or glob.glob(f"{ROOT}/{ds}/results/{f}.csv"))[0])


# ---- Figure 2: cost per resolved request relative to always-frontier, by caller
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, (ds, title) in zip(axes, DS):
    t = res(ds, "policies_by_profile")
    y = np.arange(len(POL))[::-1]
    for k, prof in enumerate(COL):
        s = t[t.profile == prof].set_index("policy")["cost_per_resolved_mean"]
        rel = np.array([s[p] / s["always_frontier"] for p, _ in POL])
        ax.scatter(rel, y + (1 - k) * 0.22, s=34, marker=MARK[prof], color=COL[prof], edgecolor=SURF,
                   linewidth=0.8, zorder=3, label=LABEL[prof])
    ax.axvline(1, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log"); ax.set_xlim(0.08, 12)
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "10"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Cost per resolved request\nrelative to always frontier (log scale)", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in POL])
    for yy in y: ax.axhline(yy - 0.5, color=GRID, lw=0.4, zorder=0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/figure2_cost_by_caller.{ext}", dpi=300, facecolor=SURF)

print("ok")
