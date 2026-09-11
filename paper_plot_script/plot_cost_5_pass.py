#!/usr/bin/env python3
"""What one pass costs: the three manager arms against Fable 5, LCB-100 x 5 passes."""
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
FILLS = {f"{k}_{arm}": palette.FILLS[k][i]
         for k in ("q38", "luna", "terra", "fable")
         for i, arm in ((0, "single"), (1, "multi"))
         if not (k == "fable" and arm == "multi")}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R4 = f"{ROOT}/runs/4models-1pass-reason-on/results"
TOKENS = os.path.join(ROOT, "runs/per_problem_tokens.json")
PASSES = [1, 2, 3, 4, 5]

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
CAP = 128_000

SOURCES = ("List rates: Qwen3.8-27B \\citep{openrouter2026}, "
           "GPT-5.6-Luna and GPT-5.6-Terra \\citep{openai2026price}, "
           "Fable~5 \\citep{anthropic2026price}. OpenAI rates are the short-context "
           "tier and Anthropic's the base rates; no long-context tier, Batch "
           "discount or prompt-caching multiplier is applied.")

TOK_KEY = {"q38_single": "q38_single"}

TESTS = [
    ("q38_single", "q38_multi", "Qwen3.8-27B: manager $-$ single"),
    ("luna_single", "luna_multi", "GPT-5.6-Luna: manager $-$ single"),
    ("terra_single", "terra_multi", "GPT-5.6-Terra: manager $-$ single"),
    ("q38_multi", "fable_single", "Fable 5 single $-$ Qwen3.8-27B manager"),
    ("luna_multi", "fable_single", "Fable 5 single $-$ GPT-5.6-Luna manager"),
    ("terra_multi", "fable_single", "Fable 5 single $-$ GPT-5.6-Terra manager"),
    ("luna_multi", "terra_single", "GPT-5.6-Terra single $-$ GPT-5.6-Luna manager"),
]

RATE_RANGE = ((0.33, 2.40), (0.45, 3.20))

BAR_W = 0.38
PITCH = 1.5
X = {"fable_single": 0.0,
     "q38_multi": PITCH - BAR_W / 2, "q38_single": PITCH + BAR_W / 2,
     "terra_multi": 2 * PITCH - BAR_W / 2, "terra_single": 2 * PITCH + BAR_W / 2,
     "luna_multi": 3 * PITCH - BAR_W / 2, "luna_single": 3 * PITCH + BAR_W / 2}
XLIMC = (0, 95)
XTICKS = [0, 20, 40, 60]
VS_X = 82

TITLE = ("What one pass costs — LCB-100, 5 passes, "
         "single call vs manager, against Fable 5")

COST_MARGINS = dict(MARGINS, left=0.025, right=0.98, top=0.842, bottom=0.206)
LEGEND_Y = 0.015
FIGSIZE_H = (FIGSIZE[0], FIGSIZE[0] * 0.86)

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
    return f"\\${v:.2f}" if v >= 0.1 else f"\\${v:.3f}"


def arm_short(key, arm):
    return (f"{arm['label'].split(',')[0]} "
            f"{'single' if key.endswith('single') else 'manager'}")


def notes(stats):
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
        a = np.array(tok[key]["tokens"], float)
        assert qids is None or tok[key]["qids"] == qids, f"{key}: different problems"
        qids = tok[key]["qids"]
        if key == "q38_single":
            a = a.copy()
            a[:, :, 1] = np.minimum(a[:, :, 1], CAP)
            a[:, :, 3] = np.minimum(a[:, :, 3], CAP)
        tin, tout = a[:, :, 0] + a[:, :, 2], a[:, :, 1] + a[:, :, 3]
        cost = (tin * ri + tout * ro) / 1e6
        passed = np.array([[bool(r["passed"]) for r in
                            json.load(open(pattern % p))["lcb"]["records"]] for p in PASSES])
        per_pass = cost.sum(axis=1)
        arms[key] = dict(
            key=key, label=label, rate=(ri, ro), cost=cost, per_pass=per_pass,
            mean=per_pass.mean(), ci=pass_ci(per_pass),
            mtok_in=tin.sum(axis=1).mean() / 1e6, mtok_out=tout.sum(axis=1).mean() / 1e6,
            per_problem=cost.mean(axis=0),
            acc=100 * passed.mean(), per_solve=cost.sum() / passed.sum(),
        )

    from scipy import stats as sps
    tests = []
    for ka, kb, name in TESTS:
        x, y = arms[ka]["per_pass"], arms[kb]["per_pass"]
        welch = sps.ttest_ind(y, x, equal_var=False)
        d = arms[kb]["per_problem"] - arms[ka]["per_problem"]
        delta, p_prob, floored = perm_sign_p(d)
        tests.append(dict(a=ka, b=kb, name=name, delta_pass=y.mean() - x.mean(),
                          delta_prob=delta, prob_ci=boot_ci(d),
                          p_pass=welch.pvalue, p_prob=p_prob, floored=floored))
    for tst, ph in zip(tests, holm([t["p_pass"] for t in tests])):
        tst["p_pass_holm"] = ph
    for tst, ph in zip(tests, holm([t["p_prob"] for t in tests])):
        tst["p_prob_holm"] = ph
        tst["sig"] = ph < ALPHA

    crossover = arms["fable_single"]["cost"].sum() / arms["q38_multi"]["cost"].sum()
    return dict(arms=arms, tests=tests, crossover=crossover)


RATE_HEADER = ("Arm", "Rate \\$/MTok in / out", "In (MTok)", "Out (MTok)",
               "\\$/pass", "\\$/solved")
RATE_SPEC = ("@{}l@{\\hspace{0.7em}}l@{\\hspace{0.7em}}r@{\\hspace{0.7em}}r"
             "@{\\hspace{0.7em}}r@{\\hspace{0.7em}}r@{}")

TABLE_HEADER = ("Comparison", "$\\Delta$ \\$/pass",
                "$p$ (per pass, $n=5$)", "$p$ (per problem, $n=100$)")
TABLE_SPEC = "@{}l@{\\hspace{1em}}c@{\\hspace{1em}}c@{\\hspace{1em}}c@{}"


def fmt_p_tex(p, floored=False):
    lt = "<" if floored else ""
    if p >= 1e-3:
        return f"${lt}{p:#.2g}$" if lt else f"{p:#.2g}"
    mantissa, exponent = ("%.1e" % p).split("e")
    return f"${lt}{mantissa}\\times10^{{{int(exponent)}}}$"


def rate_tex(v):
    return f"\\${v:g}" if v == int(v) else f"\\${v:.2f}"


def rate_rows(stats):
    rows = []
    for key, arm in stats["arms"].items():
        ri, ro = arm["rate"]
        rows.append((arm_short(key, arm), f"{rate_tex(ri)} / {rate_tex(ro)}",
                     f"{arm['mtok_in']:.4f}", f"{arm['mtok_out']:.4f}",
                     money(arm["mean"]), money(arm["per_solve"])))
    return rows


def table_rows(stats):
    def shown(key):
        return round(stats["arms"][key]["mean"], 2)
    return [(t["name"], f"{shown(t['b']) - shown(t['a']):+.2f}",
             fmt_p_tex(t["p_pass_holm"]), fmt_p_tex(t["p_prob_holm"], t["floored"]))
            for t in stats["tests"]]


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

    ax.set(xlim=XLIMC, ylim=(3 * PITCH + 0.62, -1.55))
    ax.set_yticks([])
    ax.set_xlabel("Cost of one pass (USD)", fontsize=FS_BODY, color=t["ink2"],
                  loc="left")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    ax.set_xticks(XTICKS, [f"\\${v:g}" for v in XTICKS])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])
    ax.annotate("vs Fable 5", xy=(VS_X, -1.05), ha="left", va="center",
                fontsize=FS_NOTE, color=t["muted_text"])

    for mk in ("q38", "luna", "terra", "fable"):
        y0 = X[f"{mk}_single"] - (BAR_W / 2 if f"{mk}_multi" in FILLS else 0)
        ax.annotate(arms[f"{mk}_single"]["label"].split(",")[0],
                    xy=(0, y0 - BAR_W), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FS_BODY, fontweight="bold", color=t["ink"])

    for key, arm in arms.items():
        y, v, (lo, hi) = X[key], arm["mean"], arm["ci"]
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
        ax.annotate(f"· {arm['acc']:.1f}%", xy=(hi, y),
                    xytext=(5 + 0.62 * FS_BODY * (len(price) - 1), 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_NOTE, color=t["muted_text"])

    vs_fable = [x for x in stats["tests"] if x["b"] == "fable_single"]
    for x_t in vs_fable:
        mark = stars(x_t["p_pass_holm"]) or "n.s."
        ax.annotate(mark, xy=(VS_X, X[x_t["a"]]),
                    ha="left", va="center", fontsize=FS_NOTE,
                    color=t["ink2"] if x_t["p_pass_holm"] < ALPHA else t["muted_text"])

    ax.annotate("Same 100 problems, 5 passes per condition; bars are the mean pass,\n"
                "CI across the 5; the figure after each bar is its pass@1.",
                xy=(1, 0), xycoords="axes fraction", xytext=(0, -30),
                textcoords="offset points", ha="right", va="top", linespacing=1.35,
                fontsize=FS_NOTE, color=t["muted_text"])

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
    for theme in ("light",):
        draw(stats, theme, save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
