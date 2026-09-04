#!/usr/bin/env python3
"""How close do the manager arms get to Fable 5? LCB-100 x 5 passes.

Four conditions, one problem set (escalation/lcb100_hardest_v6.json, same 100 ids in the
same order), 5 independent passes each, 128k max tokens, reasoning ON everywhere:

    Qwen3.8-27B    with manager   runs/4models-1pass-reason-on/results/q38_multiagent_p{1..5}
    GPT-5.6-Luna   with manager   runs/4models-1pass-reason-on/results/luna_multiagent_p{1..5}
    GPT-5.6-Terra  with manager   runs/4models-1pass-reason-on/results/terra_multiagent_p{1..5}
    Fable-5        single call    runs/fable5-5pass-single/results/fable5_single_p{1..5}

Every bar is a manager arm except the reference, so the scaffold is held fixed and the
chart asks one question: how much of Fable 5's single call does the manager buy on each
base model?

NO SINGLE-CALL ARM IS DRAWN except Fable 5's, which is the reference and has no other.
Qwen3.8-27B's was generated at a 250,000-token output cap against the 128,000 everything
here ran at -- 150 of its 500 calls ran past 128k and 124 stopped dead on the 250k
ceiling, so a manager-vs-single Delta drawn against it is partly a comparison of output
budgets. capmatch_q38.py replays those generations against 128k if the within-model Delta
is wanted back; *.cap128k.regraded.json holds the result.

WHAT IS STILL NOT MATCHED IS THINKING DEPTH, and it is the live caveat on this figure.
The cap is the same and the scaffold is the same, but each provider exposes a different
control over how long the model thinks and none of these four sit on the same setting:
the Qwen chat template leaves thinking on with no budget requested, bounded only by the
128k cap; Fable 5 is adaptive-always-on at effort:high; and the two GPT-5.6 arms ran at
OpenAI's DEFAULT effort, because run_4models_1pass_reason_on.sh never sets
ESCALATION_OPENAI_REASONING. It shows in the generations -- a manager pass emits 185k
output tokens per problem on Qwen3.8-27B against 10.5k on Luna and 7.9k on Terra. Read a
low GPT bar as "this model at the effort the provider defaults to", not as the model's
ceiling.

Fable 5 has NO manager arm -- that run was commissioned to measure the model, not the
scaffold (see run_fable5_5pass_single.sh) -- so no Delta here is a within-model Delta.

Bars only. The dot twin drew the same numbers in the same left-to-right order and was a
second rendering rather than a second view; the CIs it carried are on the bars.

    uv run --with matplotlib --with numpy --with scipy python \\
        paper_plot_script/plot_q38_vs_fable5_5_pass.py

Writes paper/plots/<title-slug>_bars_light.pdf (the "_bars" suffix is what
make_figures_tex.py's panels=("bars",) entry looks for).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_16k_reason_off_5_pass import (
    ALPHA, CI_LW, EDGE_LW, FIGSIZE, FS_BODY, FS_HEAD, FS_NOTE, FS_TITLE,
    MARGINS, PLOTS, THEMES,
    apply_theme, boot_ci, fmt_p_num, holm, model_block, pass_ci, perm_sign_p, ring,
    side_panel, slug, wrap_title, write_figure,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)             # repo root, one level up from paper_plot_script/
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
PASSES = [1, 2, 3, 4, 5]

# key -> (label, results file pattern). Order is the order they are drawn, left to right.
#
# These read the *.regraded.json files, not the originals. LiveCodeBench's stdin mock had a
# stateless buffer.readline() that returned line 1 on every call, so any solution reading
# multi-line input through it scored wrong however correct it was (escalation/regrade.py;
# paper S3.3).
#
# The exposure tracks how often a model writes that idiom, which is a matter of coding
# style, so the re-score is NOT a constant offset across these arms -- it is the largest
# single correction in the figure and it lands almost entirely on one bar:
#
#     Fable-5        0% of solutions use buffer.readline    +0.0 pp
#     Qwen3.8-27B    3%                                     +2.0
#     GPT-5.6-Luna  12%                                     +7.4
#     GPT-5.6-Terra 18%                                    +12.4   <- 72.6 -> 85.0
#
# Every flip is a fail becoming a pass; nothing goes the other way. Point these back at the
# plain .json and the chart silently reverts to the buggy grader, which would show Terra's
# manager arm 12 points lower for reasons that have nothing to do with the scaffold.
ARMS = {
    "q38_multi": ("Qwen3.8-27B, with manager", f"{R4}/q38_multiagent_p%d.regraded.json"),
    "luna_multi": ("GPT-5.6-Luna, with manager", f"{R4}/luna_multiagent_p%d.regraded.json"),
    "terra_multi": ("GPT-5.6-Terra, with manager", f"{R4}/terra_multiagent_p%d.regraded.json"),
    "fable_single": ("Fable-5, single call",
                     f"{ROOT}/runs/fable5-5pass-single/results/fable5_single_p%d.regraded.json"),
}

# Each manager arm against the reference, on the same 100 problems: (key a, key b, name).
# Delta is b - a, so a positive number means Fable 5 is still ahead.
TESTS = [
    ("q38_multi", "fable_single", "Fable-5 − Qwen3.8-27B + manager"),
    ("luna_multi", "fable_single", "Fable-5 − GPT-5.6-Luna + manager"),
    ("terra_multi", "fable_single", "Fable-5 − GPT-5.6-Terra + manager"),
]

# One x per model.
X = {"q38_multi": 0.0, "luna_multi": 1.0, "terra_multi": 2.0, "fable_single": 3.0}

# hue = model. Qwen3.8-27B keeps the teal it has wherever both its arms are drawn, and
# Fable-5 the magenta the reasoning-ON chart gives Opus-5, both being Anthropic models --
# those two never appear together, so the hue is free to mean "the Anthropic arm" in each.
# The GPT-5.6 pair takes bronze and blue, unused by anything sharing a figure with them.
# Every manager arm takes its model's DARK step, as it does in the sibling charts, so a
# reader arriving from one of those still reads dark = with manager.
FILLS = {
    "q38_multi": "#10605a",        # deep teal (shared palette; see plot_4new FILLS)
    "luna_multi": "#2f6fbf",       # blue
    "terra_multi": "#3b7f26",      # deep green
    "fable_single": "#e6a7a0",     # salmon; the reference, and the one arm that is not a manager
}
FABLE_DARK = "#a8306a"      # only for the reference line and Fable-5's ring

TITLE = ("The manager arms against Fable-5 "
         "— LCB-100, 5 passes, 128k max tokens, reasoning ON")

CAPTION = ("\\textbf{How much of the gap does the scaffold buy?} 128k $\\times$ 5 passes, "
           "reasoning on, three models behind a manager against Fable~5's single call.")

BAR_W = 0.46
YLIM = 104          # headroom for the value labels over the tallest bar, and no more:
                    # with the cross-model bracket gone there is nothing else up there

# Four blocks share the width two used to, and the right column names arms in full, so
# the axes give the width back through `right` and the deeper block stack through `bottom`.
PANEL_X = 0.725
Q38_MARGINS = dict(MARGINS, right=0.70, bottom=0.315)


def notes(stats):
    a, tests = stats["arms"], stats["tests"]
    gaps = ", ".join(f"{stats['short'][t['a']]} {t['delta']:+.1f} pp "
                     f"(p {fmt_p_num(t['p_holm'], 2, t['floored'])})" for t in tests)
    return [
        "Bars are pass@1 over the same 100 problems; the line through each is the 95% CI "
        "across the 5 passes (t, df = 4) — run-to-run spread, not within-pass error.",
        f"Every Δ is Fable-5 minus that manager arm, so a positive number means Fable-5 is "
        f"still ahead: {gaps}. Each is a paired sign-flip permutation test, unit = problem "
        f"(n = 100), Holm-corrected across the 3; \"<\" is the permutation floor.",
        "Nothing here is a within-model Δ. Fable-5 has no manager arm — that run was "
        "commissioned to measure the model, not the scaffold — and no single-call arm is "
        "drawn: Qwen3.8-27B's was generated at a 250k output cap against the 128k every bar "
        "here ran at, so a manager-vs-single Δ against it would be partly a difference of "
        "output budgets rather than of scaffolds.",
        "The output cap and the scaffold are matched across these four; thinking depth is "
        "not, and it is the caveat that survives. Each provider exposes a different control "
        "and no two of these sit on the same setting — the Qwen chat template leaves "
        "thinking on with no budget requested, bounded only by the 128k cap; Fable-5 is "
        "adaptive-always-on at effort:high; the two GPT-5.6 arms ran at OpenAI's default "
        "effort. A manager pass emits 185k output tokens per problem on Qwen3.8-27B against "
        "10.5k on GPT-5.6-Luna and 7.9k on GPT-5.6-Terra, so a low bar here is that model at "
        "the effort its provider defaults to, not its ceiling.",
        "Non-empty completions per 100: "
        + ", ".join(f"{stats['short'][k]} {v['ne']:.0f}" for k, v in a.items())
        + ". Empty or truncated output scores as a fail. Qwen3.8-27B ran on local vLLM "
        "behind the litellm proxy — the records name the route (groq:small-model), not a "
        "hosted endpoint — the GPT-5.6 arms on OpenAI and Fable-5 on the Anthropic API. No "
        "OpenRouter anywhere, so no silent provider swap sits inside a condition.",
    ]


# --------------------------------------------------------------------------- data

def load_arm(pattern):
    """-> dict(passed[P, N] bool, nonempty[P, N] bool, n_length, n_refusal, qids)."""
    qids, passed, nonempty, finish = None, [], [], []
    for p in PASSES:
        recs = json.load(open(pattern % p))["lcb"]["records"]
        ids = [r["question_id"] for r in recs]
        if qids is None:
            qids = ids
        assert ids == qids, f"{pattern % p}: question_id order differs"
        passed.append([bool(r["passed"]) for r in recs])
        nonempty.append([bool((r.get("code") or "").strip()) for r in recs])
        finish += [r.get("finish_reason") for r in recs]
    return dict(qids=qids, passed=np.array(passed), nonempty=np.array(nonempty),
                n_length=finish.count("length"), n_refusal=finish.count("refusal"))


def compute():
    arms = {}
    qids = None
    for key, (label, pattern) in ARMS.items():
        a = load_arm(pattern)
        assert qids is None or a["qids"] == qids, f"{key}: covers different problems"
        qids = a["qids"]
        arms[key] = dict(
            a, key=key, label=label,
            score=100 * a["passed"].mean(),
            ci=pass_ci(100 * a["passed"].mean(axis=1)),   # across the 5 passes
            prob=a["passed"].mean(axis=0),                # per-problem rate over 5 passes
            ne=100 * a["nonempty"].mean(),
        )

    tests = []
    for ka, kb, name in TESTS:
        d = arms[kb]["prob"] - arms[ka]["prob"]
        delta, p_perm, floored = perm_sign_p(d)
        tests.append(dict(a=ka, b=kb, name=name, delta=100 * delta,
                          delta_ci=tuple(100 * v for v in boot_ci(d)),
                          p_perm=p_perm, floored=floored))
    for tst, ph in zip(tests, holm([tst["p_perm"] for tst in tests])):
        tst["p_holm"] = ph
        tst["sig"] = ph < ALPHA

    # "Qwen3.8-27B, with manager" -> "Qwen3.8-27B": the model name alone, for notes that
    # list every arm and would otherwise repeat "with manager" four times
    short = {k: v["label"].split(",")[0] for k, v in arms.items()}
    return dict(arms=arms, tests=tests, short=short)


# --------------------------------------------------------------------------- plot

def draw(stats, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**Q38_MARGINS)

    arms = stats["arms"]
    by_a = {tst["a"]: tst for tst in stats["tests"]}

    ax.set(xlim=(-0.62, 3.62), ylim=(0, YLIM))
    ax.set_xticks([0, 1, 2, 3], [""] * 4)   # names live in the per-model block below
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)  # no background gridlines; every mark already carries a printed value label
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    # ---- Fable-5's score as a rule across the panel: the thing the managers are chasing.
    # With the residuals down at 1-10 pp the bar tops alone do not read as ordered, and
    # the rule is what makes each manager arm read as sitting under it.
    fable = arms["fable_single"]["score"]
    ax.axhline(fable, ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK, alpha=0.85, zorder=2)

    # ---- one bar per condition on its model's centre line
    for key, arm in arms.items():
        x, y, (lo, hi) = X[key], arm["score"], arm["ci"]
        ax.bar(x, y, BAR_W, color=FILLS[key], zorder=3,
               edgecolor=ring(FILLS[key], theme), linewidth=EDGE_LW)
        # bar runs the full interval on top of the mark, so an interval narrower than the
        # marker stays visible
        ax.plot([x, x], [lo, hi], lw=CI_LW, color=t["ink"],
                solid_capstyle="butt", zorder=5)
        for end in (lo, hi):
            ax.plot([x], [end], marker="_", ms=14, mew=CI_LW, color=t["ink"], zorder=5)
        ax.annotate(f"{y:.1f}", xy=(x, hi), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_BODY, color=t["ink"])

    # ---- blocks under the axis. Four groups on a ~4in axis, so each line has to fit in a
    # quarter of it: no size row (three of the four are undisclosed, so the row said
    # nothing and was the widest thing here), and the residual is "gap" rather than
    # "Δ vs Fable-5", which at this pitch ran into the block beside it.
    # 2 significant figures on p, not the 1 the sibling charts use: Holm multiplies the
    # permutation floor 5e-6 by 3 here, and "<1x10^-5" rounded from 1.5e-5 would claim a
    # bound the test does not support.
    drop = 0
    for key, arm in arms.items():
        rows = [(stats["short"][key], FS_HEAD, "bold", t["ink"])]
        tst = by_a.get(key)
        if tst is None:
            rows.append(("reference", FS_NOTE, "normal", t["ink2"]))
        else:
            col = t["ink2"] if tst["sig"] else t["muted_text"]
            rows += [(f"gap {-tst['delta']:+.1f} pp", FS_NOTE, "normal", col),
                     (f"p {fmt_p_num(tst['p_holm'], 2, tst['floored'])}",
                      FS_NOTE, "normal", col)]
        # max, not last: the reference block is a row shorter than the other three, and
        # taking whatever the loop ended on would hang the axis note inside them
        drop = max(drop, model_block(ax, X[key], rows))

    ax.annotate("Same 100 problems, 5 passes per condition; every bar but Fable-5 is a "
                "manager arm",
                xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -(drop + 12)), textcoords="offset points",
                ha="center", va="top", fontsize=FS_BODY, color=t["ink2"])

    # ---- right column: model swatches, then what the rule means
    def swatch(colour):
        return Patch(facecolor=colour, edgecolor=ring(colour, theme), linewidth=EDGE_LW)

    swatches = [swatch(FILLS[k]) for k in arms]
    names = [f"{stats['short'][k]}\n{'single call' if k == 'fable_single' else 'with manager'}"
             for k in arms]

    # label wraps: the column is ~2.4in wide on the page, one line would run off the canvas
    legend2 = [Line2D([], [], ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK,
                      label=f"Fable-5 single call\n{fable:.1f}%"),
               Line2D([], [], ls="", label="Δ under each bar is\nits gap to that line")]

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99,
                 va="top", fontsize=FS_TITLE, fontweight="bold", color=t["ink"],
                 linespacing=1.25)
    side_panel(fig, t, swatches, names, legend2, panel_x=PANEL_X,
               trend_kw=dict(handletextpad=0.7, handlelength=2.2))
    if save:
        # no bbox_inches="tight": the canvas is authored at exactly PAGE_SCALE x the size
        # it is printed, and a crop would scale the type differently per chart
        write_figure(fig, save)
    plt.close(fig)


def main():
    stats = compute()

    hdr = f"{'condition':30s} {'pass@1':>7s} {'95% CI (passes)':>18s}  nonempty"
    print(hdr)
    print("-" * len(hdr))
    for arm in stats["arms"].values():
        print(f"{arm['label']:30s} {arm['score']:7.1f}"
              f"   [{arm['ci'][0]:5.1f},{arm['ci'][1]:6.1f}]  {arm['ne']:5.1f}"
              f"   length={arm['n_length']:3d} refusal={arm['n_refusal']:3d} (of 500)")

    print("\nPaired sign-flip permutation tests (unit = problem, n = 100, Holm across 3):")
    for tst in stats["tests"]:
        pre = "<" if tst["floored"] else " "
        print(f"  {tst['name']:36s} {tst['delta']:+6.1f} pp"
              f"   [{tst['delta_ci'][0]:+5.1f},{tst['delta_ci'][1]:+6.1f}]"
              f"  p_perm {pre}{tst['p_perm']:.2e}  p_holm {tst['p_holm']:.2e}")

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):  # no dark twin -- the paper only \inputs light
        draw(stats, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_bars_{theme}.pdf"))


if __name__ == "__main__":
    main()
