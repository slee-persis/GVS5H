#!/usr/bin/env python3
"""Cost against accuracy: all seven pinned-backend arms on one axis, LCB-100 x 5 passes, 128k.

One point per arm -- four single calls and the three manager arms that exist -- so the
question the other two charts answer separately (what does the scaffold buy, what does it
cost) can be read as a single trade-off. x is dollars for one pass over the 100 problems,
log scale: the arms span two orders of magnitude and on a linear axis six of them pile up
against the left edge. A line joins each model's two arms, so the scaffold reads as a
move through the plane rather than as two unrelated dots.

QWEN3.8-27B IS THE 128k ARM ON BOTH AXES. Its single call was generated at 250k, and the
rest of the paper reports it cap-matched back to 128k (S3.2). Taking the 128k score with
the 250k bill would price a run that was never scored, and would put the point a third of
the way across the axis from where the scored generations sit -- so the output tokens are
capped at 128,000 per call to match, which moves that arm $29.08 -> $20.44 a pass.

Rates match plot_cost_5_pass's, and its caveats apply here unchanged: they are list prices,
no cached-input discount, and the Qwen figure is what those tokens would have cost rented
rather than what our own hardware cost to run. They are declared in ARMS below rather than
imported, because that chart covers whichever arms its own comparison needs while this one
is defined by covering all seven.

    uv run --with matplotlib --with numpy --with scipy python paper_plot_script/plot_cost_vs_score.py

Writes paper/plots/<title-slug>_light.pdf.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import palette

from plot_16k_reason_off_5_pass import (
    DOT, EDGE_LW, FIGSIZE, FS_BODY, FS_NOTE, FS_TITLE, MARGINS, PLOTS, THEMES,
    apply_theme, below_panel, pass_ci, ring, slug, wrap_title, write_figure,
)
from plot_cost_5_pass import SOURCES, TOKENS      # noqa: F401 -- SOURCES is read by
# make_figures_tex.py off this module, so it has to be an attribute of it. Same sheets as
# the cost chart, which is the point of importing rather than restating them.

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root, one level up from paper_plot_script/
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
RF = f"{ROOT}/runs/fable5-5pass-single/results"
PASSES = [1, 2, 3, 4, 5]
CAP = 128_000

# key -> (label, results pattern, (input rate, output rate) per MTok, model, arm).
#
# Declared here rather than imported from plot_cost_5_pass: that chart is free to narrow to
# whichever arms its own comparison needs, and this one is defined by covering ALL of them.
# Importing its ARMS coupled the two, and the single-call rows vanishing from this scatter
# is not a failure it would have reported -- it would just have drawn four points.
#
# The Qwen single row is the 128k cap-matched replay, not the 250k generation it was cut
# from; see the module docstring on why both axes have to move together.
ARMS = {
    "q38_single":   ("Qwen3.8-27B, single call",    f"{R4}/q38_single_p%d.cap128k.regraded.json",  (0.35, 2.75), "q38", "single"),
    "q38_multi":    ("Qwen3.8-27B, with manager",   f"{R4}/q38_multiagent_p%d.regraded.json",      (0.35, 2.75), "q38", "manager"),
    "luna_single":  ("GPT-5.6-Luna, single call",   f"{R4}/luna_single_p%d.regraded.json",         (0.20, 1.20), "luna", "single"),
    "luna_multi":   ("GPT-5.6-Luna, with manager",  f"{R4}/luna_multiagent_p%d.regraded.json",     (0.20, 1.20), "luna", "manager"),
    "terra_single": ("GPT-5.6-Terra, single call",  f"{R4}/terra_single_p%d.regraded.json",        (2.0, 12.0),  "terra", "single"),
    "terra_multi":  ("GPT-5.6-Terra, with manager", f"{R4}/terra_multiagent_p%d.regraded.json",    (2.0, 12.0),  "terra", "manager"),
    "fable_single": ("Fable 5, single call",        f"{RF}/fable5_single_p%d.regraded.json",       (10.0, 50.0), "fable", "single"),
}
LABEL = {"q38": "Qwen3.8-27B", "luna": "GPT-5.6-Luna", "terra": "GPT-5.6-Terra",
         "fable": "Claude Fable 5"}
FILLS = {k: palette.FILLS[k] for k in ("q38", "luna", "terra", "fable")}

TITLE = ("What accuracy costs — LCB-100, 5 passes, 128k max tokens, reasoning ON, "
         "all seven pinned-backend arms")

CAPTION = ("\\textbf{Cost against accuracy.} One point per arm; an arrow runs from the "
           "single call to the manager of each model that has both.")

# Legend under the axes, so the plane gets the whole text width: seven points spread over
# two decades of x, and a quarter of the canvas spent on a four-line legend was the widest
# thing on it. `bottom` holds the x label and the legend row below it.
SCATTER_MARGINS = dict(MARGINS, right=0.985, bottom=0.245)
LEGEND_Y = 0.115        # top of the legend row, clear of the x-axis label


def notes(pts):
    by = {p["key"]: p for p in pts}
    cheap = min(pts, key=lambda p: p["cost"])
    best = max(pts, key=lambda p: p["acc"])
    return [
        "x is dollars for one pass over the 100 problems (log scale), y is pass@1; the bar "
        "through each point is the 95% CI across the 5 passes (t, df = 4). Light fill = "
        "single call, dark = with manager; marker shape is the model.",
        f"Qwen3.8-27B's single arm is the 128k cap-matched one on BOTH axes — score from "
        f"the replay, output tokens capped at 128,000 per call to match. Priced as "
        f"generated at 250k it would sit at ¤{by['q38_single']['cost_asgen']:.2f} rather "
        f"than ¤{by['q38_single']['cost']:.2f}.",
        "Retried and "
        "discarded attempts are counted: they were generated and would be billed.",
        f"The cheapest arm is {cheap['label']} at ¤{cheap['cost']:.2f} a pass and the most "
        f"accurate is {best['label']} at ¤{best['cost']:.2f} — a "
        f"{best['cost'] / cheap['cost']:.0f}× spread in price for "
        f"{best['acc'] - cheap['acc']:+.1f} points."
    ]


# --------------------------------------------------------------------------- data

def compute():
    tok = json.load(open(TOKENS))
    pts = []
    for key, (label, pattern, (ri, ro), mk, arm) in ARMS.items():
        a = np.array(tok[key]["tokens"], float)      # [pass, problem, 4]: in, out, disc_in, disc_out
        capped = key == "q38_single"
        if capped:
            # Cap every generated attempt, not just the graded one: a retry that ran to
            # 250k would equally have stopped at 128k. Discarded output is 2.4% of this
            # arm's total, so the choice moves the point by cents either way.
            a = a.copy()
            a[:, :, 1] = np.minimum(a[:, :, 1], CAP)
            a[:, :, 3] = np.minimum(a[:, :, 3], CAP)
        cost = ((a[:, :, 0] + a[:, :, 2]) * ri + (a[:, :, 1] + a[:, :, 3]) * ro) / 1e6
        per_pass = cost.sum(axis=1)
        passed = np.array([[bool(r["passed"]) for r in
                            json.load(open(pattern % p))["lcb"]["records"]] for p in PASSES])
        acc_per_pass = 100 * passed.mean(axis=1)
        pt = dict(key=key, model=mk, arm=arm, label=label, cost=per_pass.mean(),
                  cost_ci=pass_ci(per_pass), acc=acc_per_pass.mean(),
                  acc_ci=pass_ci(acc_per_pass))
        if capped:
            raw = np.array(tok[key]["tokens"], float)
            pt["cost_asgen"] = (((raw[:, :, 0] + raw[:, :, 2]) * ri
                                + (raw[:, :, 1] + raw[:, :, 3]) * ro) / 1e6).sum(axis=1).mean()
        pts.append(pt)
    return pts


# --------------------------------------------------------------------------- plot

def draw(pts, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**SCATTER_MARGINS)
    ax.set_xscale("log")
    ax.set_xlabel("Cost of one pass over the 100 problems (USD, log scale)",
                  fontsize=FS_BODY, color=t["ink2"])
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.set(xlim=(0.25, 190), ylim=(55, 100))
    ax.grid(False)  # no background gridlines; every point's value is on the chart already
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100],
                  ["\\$0.50", "\\$1", "\\$2", "\\$5", "\\$10", "\\$20", "\\$50", "\\$100"])
    ax.minorticks_off()
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    by = {p["key"]: p for p in pts}
    # the scaffold as a move through the plane: single -> manager, same model
    for mk in ("q38", "luna", "terra"):
        s, m = by[f"{mk}_single"], by[f"{mk}_multi"]
        ax.annotate("", xy=(m["cost"], m["acc"]), xytext=(s["cost"], s["acc"]),
                    arrowprops=dict(arrowstyle="-|>", color=FILLS[mk][1], lw=1.4,
                                    alpha=0.55, shrinkA=9, shrinkB=11), zorder=2)

    for p in pts:
        light, dark = FILLS[p["model"]]
        fill = light if p["arm"] == "single" else dark
        edge = ring(fill, theme)
        ax.plot([p["cost_ci"][0], p["cost_ci"][1]], [p["acc"]] * 2, color=edge, lw=1.2,
                alpha=0.8, zorder=3)
        ax.plot([p["cost"]] * 2, [p["acc_ci"][0], p["acc_ci"][1]], color=edge, lw=1.2,
                alpha=0.8, zorder=3)
        # Marker SHAPE is the model, the same job hatch does on the bar charts: a
        # 6.9pt disc is too small to hatch, and this is the one figure with no
        # positional fallback -- x is price, y is accuracy, so neither axis names a
        # model and colour alone would leave grayscale readers with seven identical dots.
        ax.scatter([p["cost"]], [p["acc"]], s=DOT, color=fill, zorder=5,
                   marker=palette.MARKER[palette.SLOT[p["model"]]],
                   edgecolor=edge, linewidth=EDGE_LW)
        # Label above the point for managers, below for singles: the two arms of a model
        # sit on the same short arrow and their labels would otherwise overlap. Fable-5 is
        # the exception -- it lands a point above and a little right of Qwen's manager arm,
        # so a label below it prints straight through that marker; it goes to the right.
        if p["key"] == "fable_single":
            off, ha = (16, -4), "left"
        else:
            off, ha = (0, 13 if p["arm"] == "manager" else -20), "center"
        ax.annotate(f"{p['acc']:.1f}  \\${p['cost']:.2f}", xy=(p["cost"], p["acc"]),
                    xytext=off, textcoords="offset points", ha=ha,
                    va="bottom", fontsize=FS_NOTE, color=t["ink"])

    pairs, names = [], []
    for mk in ("q38", "luna", "terra", "fable"):
        light, dark = FILLS[mk]
        cols = (light,) if mk == "fable" else (light, dark)
        # the legend swatch takes the model's own marker, so the key is readable in
        # grayscale too -- an all-circle legend would name the models by hue alone
        pairs.append(tuple(
            Line2D([], [], marker=palette.MARKER[palette.SLOT[mk]], ls="", ms=12,
                   color=c, markeredgecolor=ring(c, theme),
                   markeredgewidth=EDGE_LW) for c in cols))
        names.append(LABEL[mk])

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99, va="top",
                 fontsize=FS_TITLE, fontweight="bold", color=t["ink"], linespacing=1.25)
    below_panel(fig, t, pairs, names, LEGEND_Y, ncol=len(names))
    if save:
        write_figure(fig, save)
    return fig


def main():
    pts = compute()
    hdr = f"{'arm':32} {'$/pass':>9} {'pass@1':>8}  {'$/point':>9}"
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(pts, key=lambda p: p["cost"]):
        print(f"{p['label']:32} {p['cost']:9.2f} {p['acc']:8.1f}  {p['cost']/p['acc']:9.3f}")
    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):  # no dark twin -- the paper only \inputs light
        draw(pts, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
