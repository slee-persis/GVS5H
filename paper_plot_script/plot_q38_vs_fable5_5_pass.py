#!/usr/bin/env python3
"""How close do the manager arms get to Fable 5? LCB-100 x 5 passes."""
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
ROOT = os.path.dirname(HERE)
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
PASSES = [1, 2, 3, 4, 5]

ARMS = {
    "q38_multi": ("Qwen3.8-27B, with manager", f"{R4}/q38_multiagent_p%d.patched.json"),
    "luna_multi": ("GPT-5.6-Luna, with manager", f"{R4}/luna_multiagent_p%d.patched.json"),
    "terra_multi": ("GPT-5.6-Terra, with manager", f"{R4}/terra_multiagent_p%d.patched.json"),
    "fable_single": ("Fable-5, single call",
                     f"{ROOT}/runs/fable5-5pass-single/results/fable5_single_p%d.patched.json"),
}

TESTS = [
    ("q38_multi", "fable_single", "Fable-5 − Qwen3.8-27B + manager"),
    ("luna_multi", "fable_single", "Fable-5 − GPT-5.6-Luna + manager"),
    ("terra_multi", "fable_single", "Fable-5 − GPT-5.6-Terra + manager"),
]

X = {"q38_multi": 0.0, "luna_multi": 1.0, "terra_multi": 2.0, "fable_single": 3.0}

FILLS = {
    "q38_multi": "#10605a",
    "luna_multi": "#2f6fbf",
    "terra_multi": "#3b7f26",
    "fable_single": "#e6a7a0",
}
FABLE_DARK = "#a8306a"

TITLE = ("The manager arms against Fable-5 "
         "— LCB-100, 5 passes, 128k max tokens, reasoning ON")

CAPTION = ("\\textbf{How much of the gap does the scaffold buy?} 128k $\\times$ 5 passes, "
           "reasoning on, three models behind a manager against Fable~5's single call.")

BAR_W = 0.46
YLIM = 104

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
            ci=pass_ci(100 * a["passed"].mean(axis=1)),
            prob=a["passed"].mean(axis=0),
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
    ax.set_xticks([0, 1, 2, 3], [""] * 4)
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    fable = arms["fable_single"]["score"]
    ax.axhline(fable, ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK, alpha=0.85, zorder=2)

    for key, arm in arms.items():
        x, y, (lo, hi) = X[key], arm["score"], arm["ci"]
        ax.bar(x, y, BAR_W, color=FILLS[key], zorder=3,
               edgecolor=ring(FILLS[key], theme), linewidth=EDGE_LW)
        ax.plot([x, x], [lo, hi], lw=CI_LW, color=t["ink"],
                solid_capstyle="butt", zorder=5)
        for end in (lo, hi):
            ax.plot([x], [end], marker="_", ms=14, mew=CI_LW, color=t["ink"], zorder=5)
        ax.annotate(f"{y:.1f}", xy=(x, hi), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_BODY, color=t["ink"])

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
        drop = max(drop, model_block(ax, X[key], rows))

    ax.annotate("Same 100 problems, 5 passes per condition; every bar but Fable-5 is a "
                "manager arm",
                xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -(drop + 12)), textcoords="offset points",
                ha="center", va="top", fontsize=FS_BODY, color=t["ink2"])

    def swatch(colour):
        return Patch(facecolor=colour, edgecolor=ring(colour, theme), linewidth=EDGE_LW)

    swatches = [swatch(FILLS[k]) for k in arms]
    names = [f"{stats['short'][k]}\n{'single call' if k == 'fable_single' else 'with manager'}"
             for k in arms]

    legend2 = [Line2D([], [], ls=(0, (6, 4)), lw=1.6, color=FABLE_DARK,
                      label=f"Fable-5 single call\n{fable:.1f}%"),
               Line2D([], [], ls="", label="Δ under each bar is\nits gap to that line")]

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99,
                 va="top", fontsize=FS_TITLE, fontweight="bold", color=t["ink"],
                 linespacing=1.25)
    side_panel(fig, t, swatches, names, legend2, panel_x=PANEL_X,
               trend_kw=dict(handletextpad=0.7, handlelength=2.2))
    if save:
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
    for theme in ("light",):
        draw(stats, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_bars_{theme}.pdf"))


if __name__ == "__main__":
    main()
