#!/usr/bin/env python3
"""Bar-chart twins of the three single-vs-manager charts: same numbers, same annotations.

One figure per condition, matching the dot/line originals one for one:

    plot_16k_reason_off_5_pass.py    -> 16k,  reasoning OFF, 5 passes
    plot_128k_reason_off_1_pass.py   -> 128k, reasoning OFF, 1 pass
    plot_128k_reason_on_1_pass.py    -> 128k, reasoning ON,  1 pass

Every statistic is imported from those modules rather than recomputed, so a bar can
never disagree with the dot it replaces. Only the drawing differs, in three ways:

  - x is categorical (models left to right by size). A log parameter axis means
    nothing for bars, so the size moves into the per-model block below the axis.
  - the paired-test bracket sits ABOVE each pair instead of beside it: the bars now
    fill the vertical space the side bracket used.
  - no interpolation curves. The first-to-last totals stay in the legend -- they are
    the difference between two measured models, not an artifact of the curve.

    uv run --with matplotlib --with numpy --with scipy python paper_plot_script/plot_bars.py

Writes paper/plots/<title-slug>_bars_compact_light.pdf -- three files, the (a)/(b)/(c)
panels of the combined scale-bars figure.
"""
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

BAR_W = 0.38        # each bar; the pair spans 2*BAR_W with no gap between them
CI_LW = 1.6

# --- compact: the (a)/(b)/(c) panels of one combined figure -------------------------
#
# HORIZONTAL bars, one panel per condition, the three panels set side by side at a
# third of the text width each (make_figures_tex.py places them in 0.32\linewidth
# subfigures). This is a second design, not a resize of draw():
#   - bars run left->right; the model order that was left->right by size is now
#     top->bottom by size.
#   - no y-tick gutter: at ~53mm printed width a tick column of model names would eat
#     a quarter of the panel, so each model's name sits flush-left ABOVE its bar pair.
#   - the right-column legend: replaced by a two-entry row under the axes, the same
#     key Figs 1 and 2 carry. Neutral swatches, because the convention being named is
#     the lightness step, which every hue on the chart makes in its own colour.
#   - the per-model Delta/p/group block: gone. Those numbers live in the combined
#     caption's per-panel notes, generated from `stats` there exactly as the block
#     itself was, so nothing is transcribed by hand twice.
# Sizing follows the repo's authored-at-2x convention: the canvas is twice the 1.76in
# print slot, which is what 0.32\linewidth comes to in the subfigure grid, so the FS_*
# sizes print at their stated value here like every other chart.
DESIGN_COL_IN = DESIGN_W_IN * 0.32
# The taller canvas and deeper bottom margin pay for the axis caption and the legend row
# under the axes; the plot area itself is what it has always been (0.99-0.135 of 1.04 =
# 0.99-0.30 of 1.42), so the bars keep the space they had. Both grew with the 7-9pt type
# band: at 1.16/0.22 the caption and the legend landed on top of each other.
FIGSIZE_COMPACT = (DESIGN_COL_IN * PAGE_SCALE, DESIGN_COL_IN * 1.42 * PAGE_SCALE)
# right stops short of 1.0 so the last x tick ("100", centred on the axis end) keeps its
# half-width inside the canvas; at 8pt that overhang no longer fits in 0.03.
MARGINS_COMPACT = dict(left=0.03, right=0.92, top=0.99, bottom=0.30)
LEG_Y_COMPACT = 0.015       # legend box bottom, figure fraction


# Each condition names its source module plus the things that genuinely differ between
# the three originals: y range, decimal places, whether per-arm CIs and per-model p
# exist, and what the first->last trend line is called. The notes live in the source
# module now -- one set per condition, covering both twins, rendered into the figure's
# LaTeX caption by make_figures_tex.py.
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
    """Rows top to bottom by single-call score, best first -- the order Figs 1 and 2
    use. Sorting on the SINGLE call, not the manager, keeps the baseline the reader
    ranks by identical across the three figures. `letters` (the Tukey groups) is
    reordered with it, since it is indexed positionally against `stats`.

    Each condition sorts independently, so a model's row differs between panels; the
    name above every pair is what identifies it, not its position.
    """
    order = sorted(range(len(stats)), key=lambda i: -stats[i]["single"])
    if letters is None:
        return [stats[i] for i in order]
    return [stats[i] for i in order], [letters[i] for i in order]


def draw_compact(cond, stats, letters, theme="light", save=None):
    """A narrow, legend-less horizontal-bar twin of draw() for the (a)/(b)/(c) panels
    of the combined scale-bars figure. See the FIGSIZE_COMPACT comment for the design."""
    t = THEMES[theme]
    apply_theme(t)
    stats = by_score(stats)
    fig, ax = plt.subplots(figsize=FIGSIZE_COMPACT)
    fig.subplots_adjust(**MARGINS_COMPACT)

    n = len(stats)
    pitch = 1.3     # group spacing: >1 leaves clear air between a group's bottom bar
                    # and the next model's name row
    ys = np.arange(n, dtype=float) * pitch
    dp, fills = cond["dp"], cond["fills"]

    # headroom above each pair holds the model's name; y runs downward so the
    # smallest model sits on top, preserving the old left->right size order. The
    # 0-100 x range is shared by all three panels so bar lengths compare across them.
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
        # the model name IS the legend here, flush left above its pair
        ax.annotate(st["label"], xy=(0, y - BAR_W), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FS_BODY, fontweight="bold", color=t["ink"])
        fill = dict(zip(("single", "multi"), fills[st["key"]]))
        # manager above its single call, as in Figs 1 and 2 (y runs downward)
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

    # bracket beside each pair with the graded Holm p, where a per-model test exists
    if cond["per_model_p"]:
        for y, st in zip(ys, stats):
            edges = [st[f"{arm}_ci"][1] if cond["ci"] else st[arm]
                     for arm in ("single", "multi")]
            x = max(edges) + 15.5   # clearance for the value labels
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

    # ---- the light/dark key, as Figs 1 and 2 carry it. The pair must keep the
    # light -> dark order on BOTH surfaces, since that ordering is the whole content
    # of the key; a theme swap that inverts it would say the opposite.
    pale, deep = ("#d5d4cd", "#6f6d67") if theme == "light" else ("#b8b6ae", "#5e5c57")

    def patch(colour):
        return Patch(facecolor=colour, edgecolor=ring(colour, theme), linewidth=EDGE_LW)

    # The key has to clear 1.76in of page at the 7pt floor. The swatch is what the reader
    # matches a bar against, so it gets the width first (1.8 em wide, a full em tall
    # rather than matplotlib's default 0.7); the gaps around it and between the two
    # entries are what give. The type itself does not.
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
        _, letters = tukey(stats)      # the pairwise p values go in the caption
        print(f"\n{cond['mod'].TITLE}")
        for i, st in enumerate(stats):
            extra = (f"  p_holm {st['p_holm']:.2e}" if cond["per_model_p"] else "")
            print(f"  {st['label']:13s} single {st['single']:5.1f}  multi {st['multi']:5.1f}"
                  f"  Δ {st['delta']:+6.1f}  Tukey {letters[i]}{extra}")
        for theme in ("light",):  # no dark twin -- the paper only \inputs light
            draw_compact(cond, stats, letters, theme,
                         save=os.path.join(PLOTS, f"{slug(cond['mod'].TITLE)}_bars_compact_{theme}.pdf"))


if __name__ == "__main__":
    main()
