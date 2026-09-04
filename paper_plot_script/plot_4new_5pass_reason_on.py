#!/usr/bin/env python3
"""Single call vs manager for the four pinned-backend models, LCB-100 x 5 passes at 128k, ON.

The companion to plot_16k_reason_off_5_pass.py (the paper's Figure 4) for the model set of
S2.1. Dots are pass@1 per condition with 95% CIs across the five passes; brackets carry the
per-model paired permutation test, Holm-corrected across the three models that have both
arms.

Each manager arm is ALSO tested against Fable 5's single call -- the dashed rule across the
panel -- since Fable 5 is here as the ceiling and a bar sitting just under the rule invites
the question of whether it is really under it. Those three tests are Holm-corrected as their
own family of 3, not pooled with the three within-model ones, so that they stay the same
numbers plot_q38_vs_fable5_5_pass.py reports -- that chart is no longer in the paper, but it
is still the second reading of this comparison and the two should not disagree. The sign
differs: here it is manager minus Fable 5, so a negative number means the bar is under the
rule, which is what the eye reads off the chart; that script reports Fable 5 minus the
manager.

Two things differ from the 16k chart deliberately:

  * The x axis is CATEGORICAL, not parameter count. Three of these four models have
    undisclosed sizes, so a scale axis would be three placeholder positions and one real
    one -- which invites exactly the cross-model reading the data cannot support. Models are
    ordered by single-call score instead, best first, top to bottom, which makes the shrinking manager
    gain legible without asserting anything about size.

  * Fable 5 contributes ONE arm. It was run single-only, so it has no delta and no bracket;
    it is here as the single-call ceiling the other three are measured against.

Reads the *.regraded.json files (see escalation/regrade.py and paper S3.3), and for
Qwen3.8's single arm the *.cap128k.regraded.json replay, so every arm is at 128k and every
arm is scored on the fixed evaluator -- matching the S2.1 table exactly.

    uv run --with matplotlib --with numpy --with scipy python paper_plot_script/plot_4new_5pass_reason_on.py

Writes paper/plots/<title-slug>_bars_light.pdf.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_16k_reason_off_5_pass import (
    ALPHA, CI_LW, EDGE_LW, FIGSIZE, FS_BODY, FS_NOTE, FS_STAR, FS_SUB, FS_TITLE,
    MARGINS, PLOTS, THEMES,
    apply_theme, fmt_p_num, holm, pass_ci, perm_sign_p, ring, slug, stars, wrap_title, write_figure,
)
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import binomtest

import palette

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root, one level up from paper_plot_script/
PASSES = [1, 2, 3, 4, 5]
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
RF = f"{ROOT}/runs/fable5-5pass-single/results"

# key -> (label, single pattern, manager pattern or None). Top-to-bottom order is the draw
# order; see the module docstring on why it is by single-call score rather than by size.
# Qwen3.8's single arm is the 128k cap-matched replay, not the 250k generation, so all four
# columns are at one cap (paper S3.2, "Cap-matching").
MODELS = [
    ("fable", "Claude Fable 5", f"{RF}/fable5_single_p%d.regraded.json", None),
    ("terra", "GPT-5.6-Terra", f"{R4}/terra_single_p%d.regraded.json",
                               f"{R4}/terra_multiagent_p%d.regraded.json"),
    ("luna",  "GPT-5.6-Luna",  f"{R4}/luna_single_p%d.regraded.json",
                               f"{R4}/luna_multiagent_p%d.regraded.json"),
    ("q38",   "Qwen3.8-27B",   f"{R4}/q38_single_p%d.cap128k.regraded.json",
                               f"{R4}/q38_multiagent_p%d.regraded.json"),
]

# The 250k generation the q38 replay was cut from, read so the console table can report
# what the cap costs instead of the number being carried by hand. The caption used to
# quote it too; that sentence is gone, since S2.2 already costs the cap out.
Q38_ASGEN = f"{R4}/q38_single_p%d.regraded.json"

# The palette lives in palette.py -- one hue per model for the whole paper, plus the
# hatch that carries the same identity in grayscale and under CVD. This file used to
# hold its own copy under a comment calling itself "THE model palette"; it and the
# copies in plot_cost_5_pass and plot_cost_vs_score had drifted apart from the scale
# figures', so one colour named two different models across the paper.
FILLS = {k: palette.FILLS[k] for k in ("q38", "luna", "terra", "fable")}
FABLE_DARK = FILLS["fable"][1]         # the rule at Fable 5's score, and its ring

TITLE = ("Manager vs single call, four models "
         "— LCB-100, 5 passes, 128k max tokens, reasoning ON")

CAPTION = ("\\textbf{Manager vs.\\ single call.} 128k $\\times$ 5 passes, "
           "reasoning on.")

BAR_W = 0.38
PITCH = 1.5         # group spacing: >1 leaves air between a group's bottom bar and
                    # the next model's name row
XLIM = 150          # bars live in 0-100; the zone beyond holds the within-model
                    # bracket and the vs-Fable-5 annotation column

# HORIZONTAL, SINGLE-COLUMN layout: models on y (top to bottom in single-call
# order), accuracy on x. The chart is designed for one column of the two-column
# layout, not the full text width, so it follows the compact scale-bars panels'
# conventions: the canvas is authored at 2 x the full design width (\halfplot is
# \panelplot, so this lands at \linewidth like every other chart and the FS_* sizes
# print at their stated value), each model's name sits flush-left ABOVE its bar pair
# instead of in a y-tick gutter, the within-model bracket sits just right of its pair,
# and the three cross-model comparisons are a headed annotation column against the
# right margin. Only the aspect ratio is this chart's own.
FIGSIZE_H = (FIGSIZE[0], FIGSIZE[0] * 0.86)
M4 = dict(MARGINS, left=0.025, right=0.98, top=0.842, bottom=0.158)
LEG_Y = 0.015       # legend box bottom, figure fraction
VS_X = 129          # left edge of the vs-Fable-5 annotation column, data units


def notes(stats):
    """The figure's caption, trimmed to what a reader needs to read the chart.

    Anything the marks already say (which models are n.s., that the x axis is
    categorical) or that belongs to the run rather than the figure (why Fable 5 has no
    manager arm) lives in the body text, not here.
    """
    tested = [s for s in stats if s["has_mgr"]]
    # Only the p values are given: every Δ is already printed on its bracket.
    gaps = ", ".join(f"{s['label']} {fmt_p_num(s['vs_p_holm'], 2, s['vs_floored'])}"
                     for s in tested)
    # ... and the floor is only worth explaining when a printed p actually hits it.
    floor = ' "<" is the permutation floor.' if any(s["vs_floored"] for s in tested) else ""
    agree = "; ".join(f"{s['label']} {s['mgr_only']}/{s['sgl_only']}" for s in tested)
    return [
        "Bars are pass@1 on the same 100 problems, the line through each the 95% CI across "
        "the 5 passes (t, df = 4). The top bar of each pair is the manager, dark; the "
        "single call is under it, light; hatch is the model. \"Single call\" is one call "
        "with no tools and no loop; Fable 5 ran single-only, so it has one bar, which the "
        "dashed rule carries across the chart.",
        f"The bracket beside each pair is manager − single call; the right-hand column "
        f"tests the top bar against Fable 5. "
        f"Both are paired sign-flip permutation tests, unit = problem (n = 100), "
        f"Holm-corrected within each family of 3: * p < .05, ** p < .01, *** p < .001.{floor}"
        f" The three within-model Δ all clear p < 1e-4; against Fable 5, p = {gaps}.",
        f"Manager-only vs single-only problem-passes ({tested[0]['n_pp']} per model): "
        f"{agree}; exact McNemar {fmt_p_num(max(s['mcnemar_p'] for s in tested), 1)} "
        f"or better. The gains are not compensating wins and losses.",
        "Every arm is at a 128k cap and re-scored on the corrected evaluator "
        "(Appendix C); Qwen3.8-27B's single arm is the 128k cap-matched replay of a "
        "250k generation, which Appendix B describes and Section 5.2 prices.",
    ]


# --------------------------------------------------------------------------- data

def load_arm(pattern):
    """-> (qids, passed[P, N] bool). One file per pass, same 100 ids in the same order."""
    qids, rows = None, []
    for p in PASSES:
        recs = json.load(open(pattern % p))["lcb"]["records"]
        ids = [r["question_id"] for r in recs]
        if qids is None:
            qids = ids
        assert ids == qids, f"{pattern % p}: id drift"
        rows.append([bool(r["passed"]) for r in recs])
    return qids, np.array(rows, float)


def compute():
    stats, qids0 = [], None
    for key, label, s_pat, m_pat in MODELS:
        qids, s = load_arm(s_pat)
        if qids0 is None:
            qids0 = qids
        assert qids == qids0, f"{key}: different problem set"
        st = dict(key=key, label=label, has_mgr=m_pat is not None,
                  single=100 * s.mean(), single_scores=100 * s.mean(axis=1),
                  single_ci=pass_ci(100 * s.mean(axis=1)),
                  single_prob=s.mean(axis=0))
        if key == "q38":
            _, g = load_arm(Q38_ASGEN)
            st["single_asgen"] = 100 * g.mean()
        if m_pat is not None:
            _, m = load_arm(m_pat)
            st.update(multi=100 * m.mean(), multi_scores=100 * m.mean(axis=1),
                      multi_ci=pass_ci(100 * m.mean(axis=1)),
                      multi_prob=m.mean(axis=0),
                      delta=100 * (m.mean() - s.mean()))
            # paired on the problem: mean over passes per problem, then sign-flip
            d = m.mean(axis=0) - s.mean(axis=0)
            _, p, floored = perm_sign_p(d)
            st.update(p_raw=p, floored=floored)
            # Pooled problem x pass discordants. The permutation test above says the
            # delta is real; only these say it is not compensating wins and losses.
            # Exact McNemar is the binomial test on the discordants (S3.2).
            mgr_only = int(((m == 1) & (s == 0)).sum())
            sgl_only = int(((m == 0) & (s == 1)).sum())
            st.update(mgr_only=mgr_only, sgl_only=sgl_only, n_pp=int(m.size),
                      mcnemar_p=binomtest(mgr_only, mgr_only + sgl_only, 0.5).pvalue)
        stats.append(st)
    tested = [s for s in stats if s["has_mgr"]]
    for s, p in zip(tested, holm([s["p_raw"] for s in tested])):
        s["p_holm"] = p

    # Second family: each manager arm against Fable 5's single call, same 100 problems and
    # the same paired test. Held to its own Holm family of 3 rather than pooled with the
    # within-model tests above -- see the module docstring.
    ref = next(s for s in stats if s["key"] == "fable")["single_prob"]
    for s in tested:
        d = s["multi_prob"] - ref
        obs, p, floored = perm_sign_p(d)
        s.update(vs_fable=100 * obs, vs_p_raw=p, vs_floored=floored)
    for s, p in zip(tested, holm([s["vs_p_raw"] for s in tested])):
        s["vs_p_holm"] = p
    return stats


# --------------------------------------------------------------------------- plot

def draw(stats, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE_H)
    fig.subplots_adjust(**M4)
    n = len(stats)
    ys = [i * PITCH for i in range(n)]
    ax.set(xlim=(0, XLIM), ylim=(ys[-1] + 0.62, -1.55))  # y downward: first model on
    # loc="left": the one-row legend sits centred under the axes, and a centred  # top
    # x label would land on top of it
    ax.set_xlabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"],
                  loc="left")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)  # no background gridlines; every mark already carries a printed value label
    ax.set_axisbelow(True)
    ax.set_yticks([])                                # names sit above their pairs
    ax.set_xticks(range(0, 101, 20))                 # ticks stop where the data can:
    ax.tick_params(length=0, labelsize=FS_BODY)      # the region beyond 100 is only
    for spine in ("left", "bottom"):                 # annotation, not scale
        ax.spines[spine].set_color(t["axis"])
    # the column of cross-model tests is bare numbers, so it needs a head once
    ax.annotate("vs Fable 5", xy=(VS_X, -1.05), ha="left", va="center",
                fontsize=FS_NOTE, color=t["muted_text"])

    # Fable 5's score as a rule down the panel. The per-row annotations carry the test;
    # the rule is what makes a 1-10 pp residual read as short of or past the ceiling at
    # a glance, three rows away from the bar it belongs to.
    fable = next(s for s in stats if s["key"] == "fable")["single"]
    ax.axvline(fable, ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK, alpha=0.85, zorder=2)

    for i, s in enumerate(stats):
        i = ys[int(i)]
        ax.annotate(s["label"], xy=(0, i - BAR_W), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FS_BODY, fontweight="bold", color=t["ink"])
        light, dark = FILLS[s["key"]]
        # manager sits ABOVE its single call in the group (y runs downward, so the
        # manager takes the negative offset); single is still drawn first, which keeps
        # the manager's ring on top where the two bars meet
        arms = [("single", light, BAR_W / 2 if s["has_mgr"] else 0.0)]
        if s["has_mgr"]:
            arms.append(("multi", dark, -BAR_W / 2))
        edges = []
        for arm, fill, dy in arms:
            v, lo, hi = s[arm], *s[f"{arm}_ci"]
            y = i + dy
            # bar_kw carries the model's hatch; it returns an edgecolor only when
            # there is one to ink, so an unhatched bar keeps ring()'s hairline
            bar = palette.bar_kw(s["key"], "manager" if arm == "multi" else "single",
                                 surface=t["surface"])
            bar.setdefault("edgecolor", ring(fill, theme))
            ax.barh(y, v, BAR_W, linewidth=EDGE_LW, zorder=3, **bar)
            # 95% CI across the five passes -- the error bar the request is about
            ax.plot([lo, hi], [y, y], color=ring(fill, theme), lw=CI_LW, zorder=5,
                    solid_capstyle="round")
            for cap in (lo, hi):
                ax.plot([cap, cap], [y - 0.045, y + 0.045], color=ring(fill, theme),
                        lw=CI_LW, zorder=5)
            # The labels get a surface-coloured backing: the Fable-5 rule crosses the
            # ones that land near it, and a bare label reads as struck through.
            ax.annotate(f"{v:.1f}", xy=(hi, y), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=FS_BODY, color=t["ink"], zorder=6,
                        bbox=dict(facecolor=t["surface"], edgecolor="none",
                                  boxstyle="square,pad=0.12"))
            edges.append(hi)
        if s["has_mgr"]:
            # bracket carrying the paired test, clear of both CI caps and both labels;
            # ticks point left, toward the pair it belongs to
            x = max(edges) + 13
            yl, yh = i - BAR_W / 2, i + BAR_W / 2
            ax.plot([x - 1.6, x, x, x - 1.6], [yl, yl, yh, yh],
                    color=t["muted"], lw=1.0, solid_joinstyle="miter", zorder=4)
            ax.annotate(f"{s['delta']:+.1f}  {stars(s['p_holm'])}",
                        xy=(x, i), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=FS_STAR, color=t["ink2"])
            # the cross-model comparison, one aligned column at the right margin: the
            # row already names the model, so no bracket out to Fable 5's row is needed
            mark = stars(s["vs_p_holm"]) or "n.s."
            ax.annotate(f"{s['vs_fable']:+.1f} {mark}",
                        xy=(VS_X, i), ha="left", va="center", fontsize=FS_NOTE,
                        color=t["ink2"] if s["vs_p_holm"] < ALPHA else t["muted_text"])

    # No model legend: under the axes it would sit directly beneath the x ticks and repeat
    # them word for word. What the ticks do NOT say is which fill is which arm, so the row
    # is the condition key instead -- neutral swatches, because the convention being named
    # is the lightness step, which every hue on the chart makes in its own colour.
    def swatch(colour):
        return Line2D([], [], marker="o", ls="", ms=12, color=colour,
                      markeredgecolor=ring(colour, theme), markeredgewidth=EDGE_LW)

    # The pair must keep the light -> dark order on BOTH surfaces, since that ordering is
    # the whole content of the key; a theme swap that inverts it would say the opposite.
    pale, deep = ("#d5d4cd", "#6f6d67") if theme == "light" else ("#b8b6ae", "#5e5c57")
    key = [(swatch(pale), "single call"),
           (swatch(deep), "with manager"),
           (Line2D([], [], ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK),
            f"Fable 5 single call, {fable:.1f}%")]

    fig.suptitle(wrap_title(TITLE), x=M4["left"], ha="left", y=0.99, va="top",
                 fontsize=FS_TITLE, fontweight="bold", color=t["ink"], linespacing=1.25)
    leg = fig.legend([h for h, _ in key], [n for _, n in key], loc="lower center",
                     bbox_to_anchor=(0.5, LEG_Y), ncol=len(key), frameon=False,
                     fontsize=FS_SUB, labelcolor=t["ink2"], handletextpad=0.7,
                     handlelength=2.2, columnspacing=2.8)
    fig.add_artist(leg)
    if save:
        write_figure(fig, save)
    return fig


def main():
    stats = compute()
    hdr = (f"{'model':16} {'single':>16} {'manager':>16} {'delta':>7} {'Holm p':>10}"
           f" {'vs Fable':>9} {'Holm p':>10}")
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        sci = f"{s['single']:5.1f} [{s['single_ci'][0]:4.1f},{s['single_ci'][1]:5.1f}]"
        if s["has_mgr"]:
            mci = f"{s['multi']:5.1f} [{s['multi_ci'][0]:4.1f},{s['multi_ci'][1]:5.1f}]"
            print(f"{s['label']:16} {sci:>16} {mci:>16} {s['delta']:+7.1f} "
                  f"{fmt_p_num(s['p_holm'], 2, s['floored']):>10} {s['vs_fable']:+9.1f} "
                  f"{fmt_p_num(s['vs_p_holm'], 2, s['vs_floored']):>10}")
        else:
            print(f"{s['label']:16} {sci:>16} {'- (single only)':>16} {'-':>7} {'-':>10} "
                  f"{'reference':>9} {'-':>10}")
    q38 = next(s for s in stats if s["key"] == "q38")
    print(f"\nQwen3.8-27B single as generated at 250k: {q38['single_asgen']:.1f} "
          f"({q38['single_asgen'] - q38['single']:+.1f} vs the 128k replay the chart uses)")

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):  # no dark twin -- the paper only \inputs light
        draw(stats, theme,
             save=os.path.join(PLOTS, f"{slug(TITLE)}_bars_{theme}.pdf"))


if __name__ == "__main__":
    main()
