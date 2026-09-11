#!/usr/bin/env python3
"""Bar-chart twins of the three single-vs-manager charts: same numbers, same annotations."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import plot_16k_reason_off_5_pass as p16
import plot_128k_reason_off_1_pass as p128off
import plot_128k_reason_on_1_pass as p128on
from plot_16k_reason_off_5_pass import (
    DESIGN_W_IN, EDGE_LW, FS_BODY, FS_NOTE, FS_STAR, PAGE_SCALE, PLOTS, THEMES,
    apply_theme, ring, slug, stars, tukey, write_figure,
)

BAR_W = 0.38
CI_LW = 1.6

# --- compact: the (a)/(b)/(c) panels of one combined figure -------------------------
DESIGN_COL_IN = DESIGN_W_IN * 0.32
FIGSIZE_COMPACT = (DESIGN_COL_IN * PAGE_SCALE, DESIGN_COL_IN * 1.42 * PAGE_SCALE)
MARGINS_COMPACT = dict(left=0.03, right=0.92, top=0.99, bottom=0.30)
LEG_Y_COMPACT = 0.015


CONDITIONS = [
    dict(mod=p16, fills=p16.FILLS, ylim=100, dp=1, ci=True, per_model_p=True,
         span="9B → 2.8T", xlabel="Models, in order of total parameters"),
    dict(mod=p128off, fills=p128off.FILLS, ylim=95, dp=0, ci=False, per_model_p=False,
         span="9B → 2.8T", xlabel="Models, in order of total parameters"),
    dict(mod=p128on, fills=p128on.FILLS, ylim=105, dp=0, ci=False, per_model_p=False,
         span="35B → Opus-5",
         xlabel="Models, in order of total parameters (Opus-5 last, size undisclosed)"),
]


def by_score(stats, letters=None):
    order = sorted(range(len(stats)), key=lambda i: -stats[i]["single"])
    if letters is None:
        return [stats[i] for i in order]
    return [stats[i] for i in order], [letters[i] for i in order]


def draw_compact(cond, stats, letters, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    stats = by_score(stats)
    fig, ax = plt.subplots(figsize=FIGSIZE_COMPACT)
    fig.subplots_adjust(**MARGINS_COMPACT)

    n = len(stats)
    pitch = 1.3
    ys = np.arange(n, dtype=float) * pitch
    dp, fills = cond["dp"], cond["fills"]

    ax.set(xlim=(0, 100), ylim=(ys[-1] + 0.62, -1.02))
    ax.set_yticks([])
    ax.set_xlabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    for y, st in zip(ys, stats):
        ax.annotate(st["label"], xy=(0, y - BAR_W), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FS_BODY, fontweight="bold", color=t["ink"])
        fill = dict(zip(("single", "multi"), fills[st["key"]]))
        for arm, off in (("single", BAR_W / 2), ("multi", -BAR_W / 2)):
            v = st[arm]
            ax.barh(y + off, v, BAR_W, color=fill[arm], zorder=3,
                    edgecolor=ring(fill[arm], theme), linewidth=EDGE_LW)
            label_x = v
            if cond["ci"]:
                lo, hi = st[f"{arm}_ci"]
                ax.plot([lo, hi], [y + off, y + off], lw=CI_LW, color=t["ink"],
                        solid_capstyle="butt", zorder=5)
                for end in (lo, hi):
                    ax.plot([end], [y + off], marker="|", ms=8, mew=CI_LW,
                            color=t["ink"], zorder=5)
                label_x = hi
            ax.annotate(f"{v:.{dp}f}", xy=(label_x, y + off), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=FS_BODY, color=t["ink"])

    if cond["per_model_p"]:
        for y, st in zip(ys, stats):
            edges = [st[f"{arm}_ci"][1] if cond["ci"] else st[arm]
                     for arm in ("single", "multi")]
            x = max(edges) + 15.5
            col = t["ink2"] if st["sig"] else t["muted_text"]
            drop = 1.8
            ax.plot([x - drop, x, x, x - drop],
                    [y - BAR_W / 2, y - BAR_W / 2, y + BAR_W / 2, y + BAR_W / 2],
                    lw=1.6, color=col, solid_joinstyle="miter", zorder=4)
            mark = stars(st["p_holm"])
            ax.annotate(mark or "n.s.", xy=(x, y), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=FS_STAR if mark else FS_NOTE,
                        color=t["ink"] if mark else t["muted_text"], zorder=5)

    pale, deep = ("#d5d4cd", "#6f6d67") if theme == "light" else ("#b8b6ae", "#5e5c57")

    def patch(colour):
        return Patch(facecolor=colour, edgecolor=ring(colour, theme), linewidth=EDGE_LW)

    leg = fig.legend([patch(pale), patch(deep)], ["single call", "with manager"],
                     loc="lower center", bbox_to_anchor=(0.5, LEG_Y_COMPACT), ncol=2,
                     frameon=False, fontsize=FS_NOTE, labelcolor=t["ink2"],
                     handlelength=1.8, handleheight=1.0,
                     handletextpad=0.35, columnspacing=0.6)
    fig.add_artist(leg)
    if save:
        write_figure(fig, save)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for cond in CONDITIONS:
        stats = cond["mod"].compute()
        _, letters = tukey(stats)
        print(f"\n{cond['mod'].TITLE}")
        for i, st in enumerate(stats):
            extra = (f"  p_holm {st['p_holm']:.2e}" if cond["per_model_p"] else "")
            print(f"  {st['label']:13s} single {st['single']:5.1f}  multi {st['multi']:5.1f}"
                  f"  Δ {st['delta']:+6.1f}  Tukey {letters[i]}{extra}")
        for theme in ("light",):
            draw_compact(cond, stats, letters, theme,
                         save=os.path.join(PLOTS, f"{slug(cond['mod'].TITLE)}_bars_compact_{theme}.pdf"))


if __name__ == "__main__":
    main()
