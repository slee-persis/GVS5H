#!/usr/bin/env python3
"""Single call vs manager across model scale, LCB-100 at 128k, reasoning ON.

"Single call" is one API call with no tools and no loop -- the raw model, not an agent.
On disk that arm is still named `*_single.json`, so the code keys stay `single`.

The reasoning-ON twin of plot_128k_reason_off_1_pass.py, over runs/results_think_high.
1 pass per condition, so only the cross-model Tukey HSD is shown.

Model set differs from the reasoning-OFF run: Qwen3.5-9b was dropped (unusable
under a reasoning budget) and Opus-5 has both arms here.

    uv run --with matplotlib --with numpy --with scipy python paper_plot_script/plot_128k_reason_on_1_pass.py

Writes paper/plots/<title-slug>_light.pdf.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_16k_reason_off_5_pass import (
    ALPHA, DOT, EDGE_LW, FIGSIZE, FILLS, FS_BODY, FS_HEAD, FS_NOTE, FS_TITLE,
    MARGINS, PLOTS, THEMES,
    apply_theme, boot_ci, head_label, model_block, ring, side_panel, slug,
    tukey, wrap_title, write_figure,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)             # repo root, one level up from paper_plot_script/
RESULTS = os.path.join(ROOT, "runs/results_think_high")

# Opus-5's parameter count is not public. It sits at a placeholder x to the right
# of Kimi-K3 -- far enough that its block under the axis clears Kimi's -- and is left
# out of the curves, the same treatment the reference chart used. Set a real value
# here if one is known.
OPUS_X = 1.2e13

MODELS = {
    "q35": ("Qwen3.6-35B", 35e9),
    "mm3": ("MiniMax-M3", 428e9),
    "kimi": ("Kimi-K3", 2.8e12),
    "opus": ("Opus-5", OPUS_X),
}

# Every model keeps the hue it has in the other two charts. Opus is new here: the
# reference chart's red for it (#a83a35) is indistinguishable from Qwen3.6-35b's
# brick (#a63f2e) once both are on screen, so it takes magenta instead.
FILLS = dict(FILLS, opus=("#e8a8c8", "#a8306a"))

TITLE = ("Manager vs single call across model scale "
         "— LCB-100, 1 pass, 128k max tokens, reasoning ON")

# see the note on CAPTION in plot_16k_reason_off_5_pass.py: make_figures_tex.py renders
# these into the figure's LaTeX caption, where they are set in real \footnotesize. The
# emit-rate line reads its numbers off `stats`, so it cannot drift from the run.
CAPTION = "\\textbf{Manager vs. single, reasoning on, 128k, one pass.}"


def notes(stats):
    """Figure 3 is the first of the three model-set charts, so it carries the shared
    conventions; Figures 4 and 5 refer back to it rather than repeating them."""
    emit = ", ".join(f"{st['label']} {st['ne_single']:.0f}\u2192{st['ne_multi']:.0f}"
                     for st in stats)
    return [
        # Exactly the sentence make_figures_tex states once for all three panels, so
        # it is stripped there rather than repeated. What used to follow it described
        # the size-ordered blocks under the axis of the old vertical chart, which the
        # compact panels do not draw and no longer order by.
        "Fill: light = single call (one call, no tools), dark = with manager; "
        "\u0394 = manager \u2212 single, in percentage points.",
        "128k max tokens, reasoning on \u2014 effort:high for Kimi-K3 and Opus-5, a 20k "
        "reasoning budget requested for Qwen3.6-35B and MiniMax-M3 that providers often did not "
        "honor. One pass per condition, so no per-model p.",
        f"Non-empty completions per 100 (single\u2192manager): {emit}. Empty or truncated "
        "output scores as a fail, so \u0394 partly tracks emit rate.",
    ]


# --------------------------------------------------------------------------- data

def load_arm(model, arm):
    """-> (qids, passed[N], nonempty[N]). One file per config: this run is 1 pass.

    The .regraded.json twin, for the reason in plot_16k_reason_off_5_pass.load_arm.
    """
    recs = json.load(open(f"{RESULTS}/{model}_{arm}.regraded.json"))["lcb"]["records"]
    return ([r["question_id"] for r in recs],
            np.array([bool(r["passed"]) for r in recs], float),
            np.array([bool((r.get("code") or "").strip()) for r in recs], float))


def compute():
    stats = []
    for key, (label, params) in MODELS.items():
        (q_s, s, ne_s), (q_m, m, ne_m) = load_arm(key, "single"), load_arm(key, "multiagent")
        assert q_s == q_m, f"{key}: arms cover different problems"
        stats.append(dict(
            key=key, label=label, params=params, d=100 * (m - s),
            single=100 * s.mean(), multi=100 * m.mean(),
            delta=100 * (m - s).mean(),
            delta_ci=tuple(100 * v for v in boot_ci(m - s)),
            ne_single=100 * ne_s.mean(), ne_multi=100 * ne_m.mean(),
        ))
    return stats


# --------------------------------------------------------------------------- plot

def draw(stats, letters, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)

    ax.set_xscale("log")
    ax.set_xticks([1e11, 1e12], ["100B", "1T"])
    ax.minorticks_off()
    # x padding is half a value label wide at each end, no more: the models have to sit
    # as far apart as the axis allows for the per-model blocks below to clear each other
    ax.set(xlim=(2.2e10, 2.2e13), ylim=(0, 112))
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)  # no background gridlines; every mark already carries a printed value label
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    # ---- one curve per condition, through every model including Opus
    from scipy.interpolate import PchipInterpolator
    lx = np.log10([st["params"] for st in stats])
    span = np.linspace(lx.min(), lx.max(), 300)
    totals = {}
    for arm, style in (("single", "--"), ("multi", "-")):
        ys = [st[arm] for st in stats]
        totals[arm] = ys[-1] - ys[0]
        ax.plot(10 ** span, PchipInterpolator(lx, ys)(span), ls=style, lw=1.6,
                color=t["muted"], alpha=0.9, zorder=1)

    # ---- both arms share the model's x, so each pair reads as one vertical stack.
    # Value labels go above the upper arm and below the lower one: models are 1.3in
    # apart on the page, and a label beside a dot would run into the next one along.
    for st in stats:
        x = st["params"]
        light, dark = FILLS[st["key"]]
        if st["single"] == st["multi"]:
            # coincident scores: one split marker, light half single / dark half manager
            ax.plot([x], [st["single"]], marker="o", ls="",
                    markersize=2 * math.sqrt(DOT / math.pi), fillstyle="left",
                    markerfacecolor=light, markerfacecoloralt=dark,
                    markeredgecolor=ring(dark, theme), markeredgewidth=EDGE_LW, zorder=4)
            ax.annotate(f"{st['single']:.0f}", xy=(x, st["single"]), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS_BODY, color=t["ink"])
            continue
        fill = dict(zip(("single", "multi"), (light, dark)))
        upper = "multi" if st["multi"] >= st["single"] else "single"
        for arm in ("single", "multi"):
            y = st[arm]
            ax.scatter([x], [y], s=DOT, color=fill[arm], zorder=4,
                       edgecolor=ring(fill[arm], theme), linewidth=EDGE_LW)
            dy, va = (12, "bottom") if arm == upper else (-12, "top")
            ax.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=FS_BODY, color=t["ink"])

    # ---- per-model block under the axis: name, size, delta, Tukey group. Names fit
    # here: this model set spans 3 decades, so no two sit closer than 1.3in.
    for i, st in enumerate(stats):
        drop = model_block(ax, st["params"], [
            (head_label(st), FS_HEAD, "bold", t["ink"]),
            (f"Δ {st['delta']:+.0f} pp", FS_NOTE, "normal", t["ink2"]),
            (f"group {letters[i]}", FS_NOTE, "normal", t["ink2"])])

    ax.annotate("Total parameters, log scale", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -(drop + 12)), textcoords="offset points",
                ha="center", va="top", fontsize=FS_BODY, color=t["ink2"])

    # ---- right column: one swatch pair per model, the two condition curves, notes
    pairs, names = [], []
    for st in stats:
        light, dark = FILLS[st["key"]]
        pairs.append(tuple(
            Line2D([], [], marker="o", ls="", ms=12, color=c,
                   markeredgecolor=ring(c, theme), markeredgewidth=EDGE_LW)
            for c in (light, dark)))
        names.append(st["label"])

    # label wraps: see the same block in plot_16k_reason_off_5_pass.py. "35B -> Opus-5"
    # is the longest span of the three charts and the one that overran the column.
    trend = [Line2D([], [], ls="--", lw=1.6, color=t["muted"],
                    label=f"Single call\n{totals['single']:+.0f} pts\n35B → Opus-5"),
             Line2D([], [], ls="-", lw=1.6, color=t["muted"],
                    label=f"With manager\n{totals['multi']:+.0f} pts\n35B → Opus-5")]

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99,
                 va="top", fontsize=FS_TITLE, fontweight="bold", color=t["ink"],
                 linespacing=1.25)
    side_panel(fig, t, pairs, names, trend,
               trend_kw=dict(handletextpad=0.7, handlelength=2.2))
    if save:
        # no crop: the canvas is authored at exactly PAGE_SCALE x its printed size
        write_figure(fig, save)
    return fig


def main():
    stats = compute()
    pmat, letters = tukey(stats)

    hdr = (f"{'model':13s} {'single':>7s} {'multi':>7s} {'delta':>7s} {'95% CI':>16s} "
           f"{'Tukey':>6s}  nonempty")
    print(hdr)
    print("-" * len(hdr))
    for i, st in enumerate(stats):
        print(f"{st['label']:13s} {st['single']:7.1f} {st['multi']:7.1f} {st['delta']:+7.1f}"
              f"   [{st['delta_ci'][0]:+5.1f},{st['delta_ci'][1]:+6.1f}] {letters[i]:>6s}"
              f"  {st['ne_single']:.0f} -> {st['ne_multi']:.0f}")

    names = [st["label"] for st in stats]
    print("\nTukey HSD across models on the per-problem delta:")
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            print(f"  {names[i]:12s} vs {names[j]:12s} "
                  f"gap {stats[i]['delta'] - stats[j]['delta']:+6.1f} pp"
                  f"   p {pmat[i][j]:.4g}")

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):  # no dark twin -- the paper only \inputs light
        draw(stats, letters, theme,
             save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
