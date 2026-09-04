#!/usr/bin/env python3
"""What one pass costs: the three manager arms against Fable 5, LCB-100 x 5 passes.

Dollars are tokens x rate, so the rates are the whole of the modelling assumption:

    Qwen3.8-27B   $0.35 / $2.75 per MTok   OpenRouter's hosted rate. There is NO
                                           first-party Alibaba price -- the weights
                                           shipped 2026-08-14 under Apache 2.0 and Qwen
                                           Cloud is still "coming soon" -- and the arms
                                           here ran on local vLLM, so this is what the
                                           same tokens would have cost rented, not what
                                           they cost us.
    GPT-5.6-Luna  $0.20 / $1.20 per MTok   OpenAI list, short-context tier.
    GPT-5.6-Terra $2 / $12 per MTok        OpenAI list, short-context tier.
    Fable 5       $10 / $50 per MTok       Anthropic list. Thinking bills as output, and
                                           on this model thinking is most of the output.

Two things the OpenAI price sheet offers and this does not take: the long-context tier
(2x the short rates, and it applies above 272k INPUT tokens -- the largest problem here
totals 88k of input across every call it made, so nothing reaches it), and the 10x
cached-input discount, which would need per-call cache-hit counts the transcripts do not
record. Ignoring the discount prices the GPT arms high, not low.

NO SINGLE-CALL ARM IS PRICED except Fable 5's, which is the reference and has no other.
Qwen3.8-27B's was generated at a 250,000-token output cap against the 128,000 every arm
here ran at -- the manager's own per-call maximum is exactly 128,000 across 3,235 calls,
and Anthropic hard-clamps Fable 5 to the same figure. 150 of its 500 calls ran past 128k
and 124 stopped dead on the 250k ceiling; those truncated calls alone are 59% of that
arm's bill. Pricing it beside these bars charges one arm for twice the output budget the
others were allowed. capmatch_q38.py replays the generations against 128k if the arm is
wanted back.

WHAT IS STILL NOT MATCHED IS THINKING DEPTH, and on a cost chart it is the caveat that
matters most, because thinking bills as output everywhere. The cap and the scaffold are
the same across these four; the effort knob is not. The Qwen chat template leaves
thinking on with no budget requested, bounded only by the 128k cap; Fable 5 is
adaptive-always-on at effort:high; and the two GPT-5.6 arms ran at OpenAI's DEFAULT
effort, because run_4models_1pass_reason_on.sh never sets ESCALATION_OPENAI_REASONING. A
manager pass emits 185k output tokens per problem on Qwen3.8-27B against 10.5k on Luna
and 7.9k on Terra, so part of what a cheap GPT bar reports is where that knob was left
rather than what the model costs to run this benchmark. Pin the effort and rerun to close
it.

Token counts come from runs/per_problem_tokens.json (see extract_tokens.py) and include
the calls that were retried and discarded -- those were generated and would be billed.

The y axis is linear dollars from zero, so the bars are in proportion to each other and
the manager surcharge reads as the height it is. The price of that is the bottom of the
range: Luna's single call ($0.41) and its manager arm ($1.50) are slivers against Fable
5's $61.11, and are read off their printed values and the per-arm table under the figure
rather than off the axis. It was a log axis until 2026-08-25 for exactly that reason.

The companion accuracy chart is plot_q38_vs_fable5_5_pass.py, on the same four arms and
the same 100 problems; this one answers what that accuracy cost.

    uv run --with matplotlib --with numpy --with scipy python \\
        paper_plot_script/plot_cost_5_pass.py

Writes paper/plots/<title-slug>_light.pdf. The stats go in the LaTeX table that
make_figures_tex.py builds under the panel, not on the chart.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import palette

from plot_16k_reason_off_5_pass import (
    ALPHA, CI_LW, EDGE_LW, FIGSIZE, FS_BODY, FS_HEAD, FS_NOTE, FS_TITLE, stars,
    MARGINS, PLOTS, THEMES,
    apply_theme, below_panel, boot_ci, fmt_p_num, holm, model_block, pass_ci,
    perm_sign_p, ring, slug, wrap_title, write_figure,
)
# The palette lives in palette.py; this chart keeps the flat "<model>_<arm>" keying its
# `arms` dict uses, so `"fable_multi" in FILLS` still answers "does this model have a
# manager arm?" -- Fable 5 was run single-only and has no dark twin.
FILLS = {f"{k}_{arm}": palette.FILLS[k][i]
         for k in ("q38", "luna", "terra", "fable")
         for i, arm in ((0, "single"), (1, "multi"))
         if not (k == "fable" and arm == "multi")}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root, one level up from paper_plot_script/
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
TOKENS = os.path.join(ROOT, "runs/per_problem_tokens.json")
PASSES = [1, 2, 3, 4, 5]

# key -> (label, results pattern, ($/MTok in, $/MTok out))
# The patterns are the regraded twins, as in plot_q38_vs_fable5_5_pass.ARMS -- the cost
# per solved problem divides by the pass count, so it moves with the grader too.
# Qwen3.8-27B's single call is the 128k CAP-MATCHED replay, and its tokens are capped to
# match (CAP below). Its manager twin and every other bar here ran at 128k; pricing this
# one off the 250k generation would put a single arm on the chart with twice the output
# budget of the arm beside it, and the surcharge column would then be measuring budgets.
ARMS = {
    "q38_single": ("Qwen3.8-27B, single call",
                   f"{R4}/q38_single_p%d.cap128k.regraded.json", (0.35, 2.75)),
    "q38_multi": ("Qwen3.8-27B, with manager",
                  f"{R4}/q38_multiagent_p%d.regraded.json", (0.35, 2.75)),
    "luna_single": ("GPT-5.6-Luna, single call",
                    f"{R4}/luna_single_p%d.regraded.json", (0.20, 1.20)),
    "luna_multi": ("GPT-5.6-Luna, with manager",
                   f"{R4}/luna_multiagent_p%d.regraded.json", (0.20, 1.20)),
    "terra_single": ("GPT-5.6-Terra, single call",
                     f"{R4}/terra_single_p%d.regraded.json", (2.0, 12.0)),
    "terra_multi": ("GPT-5.6-Terra, with manager",
                    f"{R4}/terra_multiagent_p%d.regraded.json", (2.0, 12.0)),
    "fable_single": ("Fable 5, single call",
                     f"{ROOT}/runs/fable5-5pass-single/results/fable5_single_p%d.regraded.json",
                     (10.0, 50.0)),
}
CAP = 128_000                       # the output cap every arm here is priced at

# Where the rates are published. One line under the panel rather than a reference on every
# row: four arms share the OpenAI sheet and two the Qwen one, so in the table the same
# three numbers repeated down a column of seven. Qwen has no first-party API price -- the
# weights are Apache 2.0 and Qwen Cloud has not opened -- so it cites the hosted gateway
# whose list rate we priced the local run at. plot_cost_vs_score imports this: the two
# figures price the same arms off the same sheets and must not drift apart.
SOURCES = ("List rates: Qwen3.8-27B \\citep{openrouter2026}, "
           "GPT-5.6-Luna and GPT-5.6-Terra \\citep{openai2026price}, "
           "Fable~5 \\citep{anthropic2026price}. OpenAI rates are the short-context "
           "tier and Anthropic's the base rates; no long-context tier, Batch "
           "discount or prompt-caching multiplier is applied.")

# The token file keys the single arms by their as-generated name; only the Qwen one needs
# its output capped, the other two never came close to 128k.
TOK_KEY = {"q38_single": "q38_single"}

# What each manager arm costs against the frontier reference. Fable 5 is named with its
# arm even though it has only one: a bare "Fable 5" reads as the model rather than as the
# single call it is here. Every p is Holm-corrected across the whole list, so adding a
# row moves the others.
TESTS = [
    ("q38_single", "q38_multi", "Qwen3.8-27B: manager $-$ single"),
    ("luna_single", "luna_multi", "GPT-5.6-Luna: manager $-$ single"),
    ("terra_single", "terra_multi", "GPT-5.6-Terra: manager $-$ single"),
    ("q38_multi", "fable_single", "Fable 5 single $-$ Qwen3.8-27B manager"),
    ("luna_multi", "fable_single", "Fable 5 single $-$ GPT-5.6-Luna manager"),
    ("terra_multi", "fable_single", "Fable 5 single $-$ GPT-5.6-Terra manager"),
    # The one comparison on the frontier that does not involve Fable 5: S2.2 reads the
    # price half of it off this row, so it is tested in the same family as the rest.
    ("luna_multi", "terra_single", "GPT-5.6-Terra single $-$ GPT-5.6-Luna manager"),
]

# the rate spread across hosted providers in the week this was priced; the cost gap
# between the manager and Fable 5 does not survive the top of it (see `crossover`)
RATE_RANGE = ((0.33, 2.40), (0.45, 3.20))

# one row per MODEL, two bars on each except Fable 5, which has only the one arm;
# PITCH > 1 leaves air between a group's bottom bar and the next model's name row
BAR_W = 0.38
PITCH = 1.5
# rows ordered by what the model's dearest arm costs, top to bottom
# manager above its single call in each group: y runs downward, so the manager arm
# takes the row above (centre - BAR_W/2) and the single call the row below
X = {"fable_single": 0.0,
     "q38_multi": PITCH - BAR_W / 2, "q38_single": PITCH + BAR_W / 2,
     "terra_multi": 2 * PITCH - BAR_W / 2, "terra_single": 2 * PITCH + BAR_W / 2,
     "luna_multi": 3 * PITCH - BAR_W / 2, "luna_single": 3 * PITCH + BAR_W / 2}
# Linear dollars from zero. The top is well above the dearest bar ($61.11): the two label
# lines over Fable 5 are drawn in points, about a fifth of the axis at this height, so at a
# tighter ceiling they run into the title.
#
# What this costs, and it is the whole reason the axis used to be log: the cheapest bar and
# the dearest are 150x apart, so GPT-5.6-Luna's single call ($0.41) is half a point tall and
# its manager arm ($1.50) not much more. Those two are read off their printed values and
# off the table under the figure, not off the axis. What the linear axis buys back is that
# the gaps between bars are now proportional to the dollars -- the manager surcharge on
# Qwen3.8-27B really does look like twice the height of its single call.
XLIMC = (0, 95)     # bars live in 0-70; beyond sits the vs-Fable-5 annotation column
XTICKS = [0, 20, 40, 60]
VS_X = 82           # left edge of that column, data units (dollars)

TITLE = ("What one pass costs — LCB-100, 5 passes, "
         "single call vs manager, against Fable 5")

# Legend under the axes rather than in a column beside them, so seven bars over three
# decades get the full text width. `bottom` pays for it: the stack under the axis is the
# model blocks, the two-line note, then the legend row. The cost-per-solve list that used
# to sit under the legend is a column of the first table now -- four more lines of chart
# would not fit, and the figures compare better beside the price they divide.
COST_MARGINS = dict(MARGINS, left=0.025, right=0.98, top=0.842, bottom=0.206)
LEGEND_Y = 0.015        # legend box bottom, figure fraction
# HORIZONTAL layout, matching plot_4new_5pass_reason_on: models on y, dollars on x.
# The canvas is authored at 2 x the full design width -- \halfplot is \panelplot, so
# this lands at \linewidth like every other chart and the FS_* sizes print at their
# stated value; only the aspect ratio is this chart's own.
# Each model's name sits above its pair; the manager-vs-Fable-5 tests are a headed
# column at the right margin. The rates the dollars were computed at are NOT on the
# chart -- they are a column of the first table under it (mod.TABLES), next to the
# tokens they multiply, and SOURCES cites where each is published.
FIGSIZE_H = (FIGSIZE[0], FIGSIZE[0] * 0.86)

# One line, as the figure's whole caption: the tables under the panel carry the numbers,
# so the prose only has to say what is priced and how it was tested. The rates used to be
# listed here; they are a column of the first table now, next to the tokens they multiply.
# No em dash -- the paper set parenthetical dashes as plain hyphens (commit 1686347).
CAPTION = ("\\textbf{The scaffold's bill.} Cost of one pass over the same 100 problems; "
           "Table~\\ref{tab:cost} is the arithmetic behind every bar. No cached-input "
           "discount is "
           "taken, and Qwen3.8-27B is priced at OpenRouter market rates. The top bar of "
           "each pair is the manager, dark; the single call is under it, light; all are at "
           "a 128k output cap, and hatch is the model. The "
           "right-hand column tests each top bar against Fable 5's single call "
           "(Welch per run, $n=5$ vs "
           "5; Holm-corrected across seven comparisons): * $p<.05$, ** $p<.01$, "
           "*** $p<.001$.")


def money(v):
    """$ to two decimals, or to three below a dime: a problem Luna solved rounds to
    \\$0.01 at two, which loses the difference between it and the arms around it."""
    return f"\\${v:.2f}" if v >= 0.1 else f"\\${v:.3f}"


def arm_short(key, arm):
    """'GPT-5.6-Luna, with manager' -> 'GPT-5.6-Luna manager'.

    The model name is never abbreviated: two of the four arms are GPT-5.6 models, so
    "Luna manager" alone would name a different thing depending on which figure the
    reader came from.
    """
    return (f"{arm['label'].split(',')[0]} "
            f"{'single' if key.endswith('single') else 'manager'}")


def notes(stats):
    """The caption's note block: only what neither the chart nor Table 1 already says.

    The three manager-vs-Fable-5 tests are on the brackets, every $/pass and $/solved
    figure is a column of Table 1, and each delta is the difference of two bars. What is
    left is the four comparisons that carry no bracket, and the rate sensitivity.
    """
    a, t = stats["arms"], stats["tests"]
    shown = lambda k: round(a[k]["mean"], 2)
    cross = next(x for x in t if x["a"] == "luna_multi" and x["b"] == "terra_single")
    within = [x for x in t if x["b"] != "fable_single" and x is not cross]
    worst = max(x["p_pass_holm"] for x in within)
    return [
        f"Unbracketed comparisons: each manager $-$ single increase is significant "
        f"({fmt_p_tex(worst)} or better per run, "
        f"{fmt_p_tex(within[0]['p_prob_holm'], within[0]['floored'])} per problem); "
        f"GPT-5.6-Luna's manager undercuts GPT-5.6-Terra's single call by "
        f"\\${shown(cross['b']) - shown(cross['a']):.2f} "
        f"({fmt_p_tex(cross['p_pass_holm'])} per run, "
        f"{fmt_p_tex(cross['p_prob_holm'], cross['floored'])} per problem).",
        f"At {stats['crossover']:.2f}x the assumed Qwen rate (\\${0.35 * stats['crossover']:.3f}/"
        f"\\${2.75 * stats['crossover']:.2f} per MTok, inside the spread across hosted "
        f"providers) the manager's cost advantage over Fable 5 disappears entirely — a "
        f"larger uncertainty than any p-value here.",
    ]

# --------------------------------------------------------------------------- data

def compute():
    tok = json.load(open(TOKENS))
    arms, qids = {}, None
    for key, (label, pattern, (ri, ro)) in ARMS.items():
        a = np.array(tok[key]["tokens"], float)          # [pass, problem, 4]
        assert qids is None or tok[key]["qids"] == qids, f"{key}: different problems"
        qids = tok[key]["qids"]
        if key == "q38_single":
            # Cap every generated attempt, not just the graded one: a retry that ran to
            # 250k would equally have stopped at 128k. This is what makes the bar match
            # the cap-matched score printed above it.
            a = a.copy()
            a[:, :, 1] = np.minimum(a[:, :, 1], CAP)
            a[:, :, 3] = np.minimum(a[:, :, 3], CAP)
        # every token generated, retried-and-discarded attempts included: all billable
        tin, tout = a[:, :, 0] + a[:, :, 2], a[:, :, 1] + a[:, :, 3]
        cost = (tin * ri + tout * ro) / 1e6
        passed = np.array([[bool(r["passed"]) for r in
                            json.load(open(pattern % p))["lcb"]["records"]] for p in PASSES])
        per_pass = cost.sum(axis=1)                      # $ for the 100 problems, per pass
        arms[key] = dict(
            key=key, label=label, rate=(ri, ro), cost=cost, per_pass=per_pass,
            mean=per_pass.mean(), ci=pass_ci(per_pass),
            # MTok over the 100 problems, averaged across the 5 passes -- the two numbers
            # that, times the rate, are `mean`
            mtok_in=tin.sum(axis=1).mean() / 1e6, mtok_out=tout.sum(axis=1).mean() / 1e6,
            per_problem=cost.mean(axis=0),               # $/problem over the 5 passes
            acc=100 * passed.mean(), per_solve=cost.sum() / passed.sum(),
        )

    from scipy import stats as sps
    tests = []
    for ka, kb, name in TESTS:
        x, y = arms[ka]["per_pass"], arms[kb]["per_pass"]
        # passes are independent runs sharing only the problem set, so the run-level
        # comparison is two-sample: pairing pass 3 with pass 3 would mean nothing
        welch = sps.ttest_ind(y, x, equal_var=False)
        d = arms[kb]["per_problem"] - arms[ka]["per_problem"]   # same Delta, /100
        delta, p_prob, floored = perm_sign_p(d)
        tests.append(dict(a=ka, b=kb, name=name, delta_pass=y.mean() - x.mean(),
                          delta_prob=delta, prob_ci=boot_ci(d),
                          p_pass=welch.pvalue, p_prob=p_prob, floored=floored))
    for tst, ph in zip(tests, holm([t["p_pass"] for t in tests])):
        tst["p_pass_holm"] = ph
    for tst, ph in zip(tests, holm([t["p_prob"] for t in tests])):
        tst["p_prob_holm"] = ph
        tst["sig"] = ph < ALPHA

    # what multiple of the assumed Qwen rate makes the manager cost what Fable 5 costs
    crossover = arms["fable_single"]["cost"].sum() / arms["q38_multi"]["cost"].sum()
    return dict(arms=arms, tests=tests, crossover=crossover)


# the tables make_figures_tex.py sets under the panel. The module owns the numbers and
# their column names; the script owns only the tabular scaffolding around them.

# 1. what each bar is made of: rate x tokens = dollars, one row per arm. The rates are the
# whole of the modelling assumption (see the module docstring), so they are a column with
# a reference on it rather than a sentence in the caption.
RATE_HEADER = ("Arm", "Rate \\$/MTok in / out", "In (MTok)", "Out (MTok)",
               "\\$/pass", "\\$/solved")
# 0.7em between columns, not the 1em the comparison table below uses: six columns at 1em
# put the header row 32pt past the text block.
RATE_SPEC = ("@{}l@{\\hspace{0.7em}}l@{\\hspace{0.7em}}r@{\\hspace{0.7em}}r"
             "@{\\hspace{0.7em}}r@{\\hspace{0.7em}}r@{}")

# 2. the gaps between those bars, tested
TABLE_HEADER = ("Comparison", "$\\Delta$ \\$/pass",
                "$p$ (per pass, $n=5$)", "$p$ (per problem, $n=100$)")
# 1em between columns rather than booktabs' 2 x \tabcolsep (12pt): naming every arm in
# full puts the longest row 2pt past the text block at the default spacing
TABLE_SPEC = "@{}l@{\\hspace{1em}}c@{\\hspace{1em}}c@{\\hspace{1em}}c@{}"


def fmt_p_tex(p, floored=False):
    """p as LaTeX. Unlike fmt_p_num's matplotlib mathtext, the "<" goes INSIDE the math:
    a bare "<" in LaTeX text mode sets an inverted exclamation mark, not a less-than."""
    lt = "<" if floored else ""
    if p >= 1e-3:
        # "#" keeps the trailing zero, so the column reads 0.0051 / 0.20, not 0.0051 / 0.2
        return f"${lt}{p:#.2g}$" if lt else f"{p:#.2g}"
    mantissa, exponent = ("%.1e" % p).split("e")
    return f"${lt}{mantissa}\\times10^{{{int(exponent)}}}$"


def rate_tex(v):
    """$0.35, but $2 -- a whole-dollar rate is published without cents and printing
    "$2.00" beside "$0.35" reads as a precision the price sheet does not claim."""
    return f"\\${v:g}" if v == int(v) else f"\\${v:.2f}"


def rate_rows(stats):
    """-> [(arm, rate + cite, MTok in, MTok out, $/pass)], already LaTeX.

    Rate x tokens is the whole calculation, so the row shows all three factors: a reader
    who disagrees with a rate can redo the $/pass column without the transcripts. That is
    what fixes the token columns at 4dp: $/pass is computed from the full-precision counts,
    so whatever the columns drop comes back multiplied by the output rate. At 2dp Fable 5
    reconstructed to $60.90 against the $61.11 printed beside it -- 4.3 ktok of rounding at
    $50/MTok -- and only one row of the seven reproduced its own dollars. 4dp is exact for
    every row; 3dp still leaves two a cent out.

    The last column is what a problem the arm got RIGHT cost, and it does not rank the
    arms the way $/pass does -- which is the chart's point, so it is worth the column. It
    used to be a list beside the chart; a column beside the price it divides is where it
    can actually be compared.
    """
    rows = []
    for key, arm in stats["arms"].items():
        ri, ro = arm["rate"]
        rows.append((arm_short(key, arm), f"{rate_tex(ri)} / {rate_tex(ro)}",
                     f"{arm['mtok_in']:.4f}", f"{arm['mtok_out']:.4f}",
                     money(arm["mean"]), money(arm["per_solve"])))
    return rows


def table_rows(stats):
    """-> [(comparison, Delta $/pass, p per pass, p per problem)], already LaTeX.

    Both p columns test the SAME difference -- the per-problem mean is the per-pass
    figure over 100 -- and differ only in what they resample: runs, or problems.

    The Delta is differenced from the two $/pass figures AS PRINTED in the table above,
    not from the full-precision means. Both tables sit on one page and a reader checks the
    second against the first: Terra's exact +8.3060 sets as +8.31 beside an 11.71 and a
    3.41 that subtract to 8.30, and Fable 5 minus Luna's manager the same way. Rounding
    twice costs a cent of a column that is only ever read to the cent; disagreeing with
    the table above it costs the reader the arithmetic. The tests themselves are on the
    full-precision per-pass costs -- only this display column rounds first.
    """
    def shown(key):
        return round(stats["arms"][key]["mean"], 2)
    return [(t["name"], f"{shown(t['b']) - shown(t['a']):+.2f}",
             fmt_p_tex(t["p_pass_holm"]), fmt_p_tex(t["p_prob_holm"], t["floored"]))
            for t in stats["tests"]]


# in the order they are set under the panel: what a pass cost, then what the differences
# between those costs test at. make_figures_tex.py reads this, not the pieces above.
# The fourth item is the table's own caption: these two sit inside a figure, so they take
# their number from \captionof{table} rather than from a table float of their own.
TABLES = [(RATE_HEADER, RATE_SPEC, rate_rows,
           "\\textbf{What one pass cost each arm.} List rate $\\times$ the tokens it "
           "consumed."),
          ]


# --------------------------------------------------------------------------- plot

def draw(stats, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE_H)
    fig.subplots_adjust(**COST_MARGINS)
    arms = stats["arms"]

    ax.set(xlim=XLIMC, ylim=(3 * PITCH + 0.62, -1.55))  # y downward
    ax.set_yticks([])                        # names live above their pairs
    # loc="left": the one-row legend sits centred under the axes, and a centred
    # x label would land on top of it
    ax.set_xlabel("Cost of one pass (USD)", fontsize=FS_BODY, color=t["ink2"],
                  loc="left")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)  # no background gridlines; every mark already carries a printed value label
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    ax.set_xticks(XTICKS, [f"\\${v:g}" for v in XTICKS])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])
    # the column of cross-model tests is bare stars, so it needs a head once
    ax.annotate("vs Fable 5", xy=(VS_X, -1.05), ha="left", va="center",
                fontsize=FS_NOTE, color=t["muted_text"])

    # ---- name row above each pair
    for mk in ("q38", "luna", "terra", "fable"):
        y0 = X[f"{mk}_single"] - (BAR_W / 2 if f"{mk}_multi" in FILLS else 0)
        ax.annotate(arms[f"{mk}_single"]["label"].split(",")[0],
                    xy=(0, y0 - BAR_W), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FS_BODY, fontweight="bold", color=t["ink"])

    # ---- one bar per condition, with the accuracy that money bought printed after the
    # price: the point of the chart is the ratio between the two, and a reader who has to
    # fetch the accuracy from the companion figure will not compute it
    for key, arm in arms.items():
        y, v, (lo, hi) = X[key], arm["mean"], arm["ci"]
        # from zero, so the length IS the price -- the axis floor the log version had to
        # draw from, and subtract back off every bar, is gone with it
        # bar_kw carries the model's hatch -- the channel that keeps the models apart
        # in grayscale; it returns an edgecolor only when there is one to ink, so an
        # unhatched bar keeps ring()'s hairline
        mk, armname = key.rsplit("_", 1)
        bar = palette.bar_kw(mk, "manager" if armname == "multi" else "single",
                             surface=t["surface"])
        bar.setdefault("edgecolor", ring(FILLS[key], theme))
        ax.barh(y, v, BAR_W, linewidth=EDGE_LW, zorder=3, **bar)
        ax.plot([lo, hi], [y, y], lw=CI_LW, color=t["ink"], solid_capstyle="butt", zorder=5)
        for end in (lo, hi):
            ax.plot([end], [y], marker="|", ms=10, mew=CI_LW, color=t["ink"], zorder=5)
        price = money(v)
        ax.annotate(price, xy=(hi, y), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_BODY, color=t["ink"])
        # its pass@1 follows in the muted step; the offset walks past the price at
        # ~0.62em per character, close enough with the air the separator dot adds
        ax.annotate(f"· {arm['acc']:.1f}%", xy=(hi, y),
                    xytext=(5 + 0.62 * FS_BODY * (len(price) - 1), 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_NOTE, color=t["muted_text"])

    # ---- the manager-vs-Fable-5 tests, one aligned column at the right margin: the
    # row already names the model, so no bracket out to Fable 5's row is needed. The
    # within-model increases are what the paired bars already show; these three are the
    # comparison the figure exists to make.
    vs_fable = [x for x in stats["tests"] if x["b"] == "fable_single"]
    for x_t in vs_fable:
        mark = stars(x_t["p_pass_holm"]) or "n.s."
        ax.annotate(mark, xy=(VS_X, X[x_t["a"]]),
                    ha="left", va="center", fontsize=FS_NOTE,
                    color=t["ink2"] if x_t["p_pass_holm"] < ALPHA else t["muted_text"])

    # two lines under the axis, right-aligned so they clear the left-loc x label
    ax.annotate("Same 100 problems, 5 passes per condition; bars are the mean pass,\n"
                "CI across the 5; the figure after each bar is its pass@1.",
                xy=(1, 0), xycoords="axes fraction", xytext=(0, -30),
                textcoords="offset points", ha="right", va="top", linespacing=1.35,
                fontsize=FS_NOTE, color=t["muted_text"])

    # ---- under the axes: the one thing the left-margin blocks cannot say -- which fill
    # is which condition. Neutral swatches, because the convention being named is the
    # lightness step, which every hue on the chart makes in its own colour.
    pale, deep = ("#d5d4cd", "#6f6d67") if theme == "light" else ("#b8b6ae", "#5e5c57")

    def patch(colour):
        return Patch(facecolor=colour, edgecolor=ring(colour, theme), linewidth=EDGE_LW)

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99,
                 va="top", fontsize=FS_TITLE, fontweight="bold", color=t["ink"],
                 linespacing=1.25)
    leg = fig.legend([patch(pale), patch(deep)], ["single call", "with manager"],
                     loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y), ncol=2,
                     frameon=False, fontsize=FS_BODY, labelcolor=t["ink2"],
                     handletextpad=0.7, columnspacing=2.8)
    fig.add_artist(leg)
    if save:
        # no crop: the canvas is authored at exactly PAGE_SCALE x its printed size
        write_figure(fig, save)
    plt.close(fig)


def main():
    stats = compute()

    hdr = (f"{'arm':26s} {'$/pass':>8s} {'95% CI':>16s} {'5 passes':>9s} "
           f"{'pass@1':>7s} {'$/solve':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in stats["arms"].values():
        print(f"{arm['label']:26s} {arm['mean']:8.2f}   [{arm['ci'][0]:5.2f},{arm['ci'][1]:6.2f}]"
              f" {arm['cost'].sum():9.2f} {arm['acc']:7.1f} {arm['per_solve']:8.3f}")

    print(f"\nsame difference, two units of analysis "
          f"(both Holm-corrected across the {len(TESTS)}):")
    for t in stats["tests"]:
        pre = "<" if t["floored"] else " "
        print(f"  {t['name'].replace('$-$', '-'):32s} {t['delta_pass']:+7.2f} $/pass"
              f"  ({t['delta_prob']:+.3f} $/problem, CI [{t['prob_ci'][0]:+.3f},"
              f"{t['prob_ci'][1]:+.3f}])")
        print(f"  {'':32s} per-pass Welch p {t['p_pass']:.2e} -> Holm {t['p_pass_holm']:.2e}"
              f"   per-problem paired p {pre}{t['p_prob']:.2e} -> Holm {t['p_prob_holm']:.2e}")
    print(f"\n  manager cost = Fable 5 cost at {stats['crossover']:.3f}x the assumed Qwen rate"
          f"  (${0.35 * stats['crossover']:.3f}/${2.75 * stats['crossover']:.2f} per MTok)")

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):  # no dark twin -- the paper only \inputs light
        draw(stats, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
