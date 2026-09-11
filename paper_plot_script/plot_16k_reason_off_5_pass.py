#!/usr/bin/env python3
"""Single call vs manager across model scale, LCB-100 x 5 passes at 16k, reasoning OFF."""
import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import palette

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "runs/models-lcb-5pass/results")
PLOTS = os.path.join(ROOT, "paper/plots")
PASSES = [1, 2, 3, 4, 5]

MODELS = {
    "q9": ("Qwen3.5-9B", 9e9),
    "q35": ("Qwen3.6-35B", 35e9),
    "mm3": ("MiniMax-M3", 428e9),
    "kimi": ("Kimi-K3", 2.8e12),
}

FILLS = {
    "q9": ("#b8aee8", "#3a35a5"),
    "q35": ("#6cb8d6", "#125f7d"),
    "mm3": ("#a8cdf0", "#2d6ab7"),
    "kimi": ("#a8d878", "#387723"),
}

TITLE = ("Manager vs single call across model scale "
         "\u2014 LCB-100, 5 passes, 16k max tokens, reasoning OFF")

CAPTION = "\\textbf{Reasoning off:} 16k $\\times$ 5 passes."


def notes(stats):
    return [
        "16k max tokens, reasoning off, five passes per condition; the line through "
        "each bar is the 95% CI across the five passes (t, df = 4).",
        "Each \u0394 is a paired sign-flip permutation test, unit = problem (n = 100), "
        "Holm-corrected across the 4 models; the bracket beside a pair grades its p: "
        "* p < .05, ** p < .01, *** p < .001, n.s. otherwise.",
    ]


N_PERM = 200_000
N_BOOT = 20_000
SEED = 42
ALPHA = 0.05
DOT = 150
EDGE_LW = 1.6
CI_LW = 1.6

PAGE_SCALE = 2.0

FS_TITLE = 9.0 * PAGE_SCALE
FS_BODY = 8.0 * PAGE_SCALE
FS_SUB = 7.5 * PAGE_SCALE
FS_NOTE = 7.0 * PAGE_SCALE
FS_HEAD = 7.0 * PAGE_SCALE
FS_STAR = 8.0 * PAGE_SCALE

TEXT_WIDTH_IN = 5.5
DESIGN_W_IN = TEXT_WIDTH_IN
FIGSIZE = (DESIGN_W_IN * PAGE_SCALE, DESIGN_W_IN * (3.1 / 6.5) * PAGE_SCALE)
MARGINS = dict(left=0.085, right=0.737, top=0.89, bottom=0.30)
PANEL_X = 0.762
PANEL_TOP = 0.89
ROW_0 = 28
ROW_PITCH = 1.35

THEMES = {
    "light": dict(
        surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        muted_text="#6b6a64", grid="#e1e0d9", axis="#c3c2b7",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        muted_text="#898781", grid="#2c2c2a", axis="#383835",
    ),
}


def write_figure(fig, save):
    stem = os.path.splitext(save)[0]
    fig.savefig(stem + ".pdf")
    print("wrote", stem + ".pdf")


def apply_theme(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["axis"], "grid.color": t["grid"],
        "axes.titlecolor": t["ink"],
        "font.family": ["Arial", "Liberation Sans", "Nimbus Sans", "Helvetica",
                        "DejaVu Sans"],
        "font.size": 10,
        "mathtext.default": "regular", "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.linewidth": 0.8,
        "figure.dpi": 130,
        "hatch.linewidth": palette.HATCH_LW * PAGE_SCALE,
    })


# --------------------------------------------------------------------------- data

def load_arm(model, arm):
    qids, passed, nonempty = None, [], []
    for p in PASSES:
        fn = f"{RESULTS}/{model}_{arm}_p{p}.patched.json"
        if not os.path.exists(fn):
            continue
        lcb = json.load(open(fn)).get("lcb")
        recs = lcb.get("records") if isinstance(lcb, dict) else None
        if not recs:
            continue
        ids = [r["question_id"] for r in recs]
        if qids is None:
            qids = ids
        assert ids == qids, f"{fn}: question_id order differs"
        passed.append([bool(r["passed"]) for r in recs])
        nonempty.append([bool((r.get("code") or "").strip()) for r in recs])
    if not passed:
        return None
    return qids, np.array(passed), np.array(nonempty)


# -------------------------------------------------------------------------- tests

def perm_sign_p(d, n_perm=N_PERM, seed=SEED):
    d = np.asarray(d, float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, d.size)) * d).mean(axis=1)
    b = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    return obs, (b + 1) / (n_perm + 1), b == 0


def boot_ci(x, n_boot=N_BOOT, seed=SEED, level=0.95):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed + 1)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return tuple(np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def pass_ci(scores, level=0.95):
    from scipy.stats import t as t_dist
    scores = np.asarray(scores, float)
    half = t_dist.ppf((1 + level) / 2, scores.size - 1) * scores.std(ddof=1) / math.sqrt(scores.size)
    return scores.mean() - half, scores.mean() + half


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        out[i] = min(1.0, running)
    return out


def compute():
    stats = []
    for key, (label, params) in MODELS.items():
        s, m = load_arm(key, "single"), load_arm(key, "multiagent")
        assert s and m, f"{key}: missing an arm"
        assert s[0] == m[0], f"{key}: arms cover different problems"
        sp, mp = s[1], m[1]
        s_prob, m_prob = sp.mean(axis=0), mp.mean(axis=0)
        delta, p_perm, floored = perm_sign_p(m_prob - s_prob)
        stats.append(dict(
            key=key, label=label, params=params, d=100 * (m_prob - s_prob),
            single=100 * sp.mean(), single_ci=pass_ci(100 * sp.mean(axis=1)),
            multi=100 * mp.mean(), multi_ci=pass_ci(100 * mp.mean(axis=1)),
            delta=100 * delta, delta_ci=tuple(100 * v for v in boot_ci(m_prob - s_prob)),
            p_perm=p_perm, floored=floored,
            ne_single=100 * s[2].mean(), ne_multi=100 * m[2].mean(),
        ))
    for st, ph in zip(stats, holm([st["p_perm"] for st in stats])):
        st["p_holm"] = ph
        st["sig"] = ph < ALPHA
    return stats


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < ALPHA else ""


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_p_num(p, sig=1, floored=False):
    prefix = "<" if floored else ""
    if p >= 1e-3:
        return prefix + f"%.{sig}g" % p
    mantissa, exponent = (f"%.{sig - 1}e" % p).split("e")
    return prefix + f"{mantissa}×10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def wrap_title(text):
    return text.replace("— ", "—\n")


def slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9-]+", "_",
                                     text.lower().replace("\u2014", " "))).strip("_")


def fmt_params(n):
    return f"{n / 1e12:g}T" if n >= 1e12 else f"{n / 1e9:g}B"


def head_label(st):
    return st["label"] if st["key"] == "opus" else fmt_params(st["params"])


def rgb(hexcolor):
    return [int(hexcolor[i:i + 2], 16) for i in (1, 3, 5)]


def ring(hexcolor, theme):
    r, g, b = rgb(hexcolor)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    f = 1 + (1 - 0.72) if (theme == "dark" and lum < 0.4) else 0.72
    return "#%02x%02x%02x" % tuple(min(255, int(c * f)) for c in (r, g, b))


def tint(hexcolor, theme, f=0.16):
    sr, sg, sb = rgb(THEMES[theme]["surface"])
    r, g, b = rgb(hexcolor)
    return "#%02x%02x%02x" % tuple(
        int(s + (c - s) * f) for s, c in ((sr, r), (sg, g), (sb, b)))


# ------------------------------------------------------------------ side panel

def _panel_height(fig, artist):
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()).height


def model_block(ax, x, rows):
    y = ROW_0
    for text, size, weight, col in rows:
        ax.annotate(text, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -y), textcoords="offset points",
                    ha="center", va="top", fontsize=size, fontweight=weight, color=col)
        y += ROW_PITCH * size
    return y


def side_panel(fig, t, swatches, names, trend, trend_kw=None, panel_x=None, panel_top=None):
    x = PANEL_X if panel_x is None else panel_x
    top = PANEL_TOP if panel_top is None else panel_top
    gap = 0.04
    leg1 = fig.legend(swatches, names, loc="upper left",
                      bbox_to_anchor=(x, top), ncol=1, frameon=False,
                      fontsize=FS_SUB, labelcolor=t["ink2"],
                      handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                      handletextpad=0.7)
    fig.add_artist(leg1)
    y = top - _panel_height(fig, leg1) - gap

    leg2 = fig.legend(handles=trend, loc="upper left", bbox_to_anchor=(x, y),
                      ncol=1, frameon=False, fontsize=FS_NOTE, labelcolor=t["ink2"],
                      **(trend_kw or {}))
    fig.add_artist(leg2)


def below_panel(fig, t, swatches, names, y, ncol, trend=None, trend_kw=None):
    leg1 = fig.legend(swatches, names, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=ncol, frameon=False, fontsize=FS_SUB, labelcolor=t["ink2"],
                      handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                      handletextpad=0.7, columnspacing=2.2)
    fig.add_artist(leg1)
    if not trend:
        return
    leg2 = fig.legend(handles=trend, loc="upper center",
                      bbox_to_anchor=(0.5, y - _panel_height(fig, leg1) - 0.02),
                      ncol=len(trend), frameon=False, fontsize=FS_NOTE,
                      labelcolor=t["ink2"], **(trend_kw or {}))
    fig.add_artist(leg2)


def tukey_sentence(stats, pmat):
    parts = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            p = pmat[i][j]
            parts.append(f"{head_label(stats[i])} vs {head_label(stats[j])} "
                         f"{fmt_p_num(p, 2)}{'' if p < ALPHA else ' n.s.'}")
    joined = "; ".join(parts)
    return "Pairwise Tukey p: " + joined + ("" if joined.endswith(".") else ".")


# ----------------------------------------------------------------- cross-model

def tukey(stats, alpha=ALPHA):
    from scipy.stats import tukey_hsd
    res = tukey_hsd(*[st["d"] for st in stats])
    p = res.pvalue
    n = len(stats)
    means = [st["delta"] for st in stats]

    nonsig = {(i, j) for i in range(n) for j in range(i + 1, n) if p[i][j] > alpha}
    cliques = []
    for mask in range(1, 1 << n):
        members = [i for i in range(n) if mask >> i & 1]
        if all((a, b) in nonsig for a in members for b in members if a < b):
            cliques.append(set(members))

    edges, seen, chosen = set(nonsig), set(), []
    while edges or len(seen) < n:
        def score(c):
            return (sum((a, b) in edges for a in c for b in c if a < b), len(c - seen))
        best = max(cliques, key=score)
        if score(best) == (0, 0):
            break
        chosen.append(best)
        edges -= {(a, b) for a in best for b in best if a < b}
        seen |= best
    chosen.sort(key=lambda c: -max(means[i] for i in c))

    letters = {i: "" for i in range(n)}
    for k, c in enumerate(chosen):
        for i in c:
            letters[i] += chr(ord("a") + k)
    return p, letters


# --------------------------------------------------------------------------- plot

def draw(stats, letters, theme="light", save=None):
    t = THEMES[theme]
    apply_theme(t)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)

    ax.set_xscale("log")
    ax.set_xticks([1e10, 1e11, 1e12], ["10B", "100B", "1T"])
    ax.minorticks_off()
    ax.set(xlim=(6e9, 4.2e12), ylim=(0, 88))
    ax.set_ylabel("Accuracy (pass@1, %)", fontsize=FS_BODY, color=t["ink2"])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=FS_BODY)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])

    def y_offset(x, y, dy_points):
        px = ax.transData.transform((x, y))
        return ax.transData.inverted().transform(
            (px[0], px[1] + dy_points * fig.dpi / 72.0))[1]

    def x_offset(x, dx_points):
        px = ax.transData.transform((x, 0))
        return ax.transData.inverted().transform(
            (px[0] + dx_points * fig.dpi / 72.0, px[1]))[0]

    gap = math.sqrt(DOT / math.pi) + EDGE_LW / 2

    from scipy.interpolate import PchipInterpolator
    lx = np.log10([st["params"] for st in stats])
    span = np.linspace(lx.min(), lx.max(), 300)
    totals = {}
    for arm, style in (("single", "--"), ("multi", "-")):
        ys = [st[arm] for st in stats]
        totals[arm] = ys[-1] - ys[0]
        ax.plot(10 ** span, PchipInterpolator(lx, ys)(span), ls=style, lw=1.6,
                color=t["muted"], alpha=0.9, zorder=1)

    for st in stats:
        x = st["params"]
        fill = dict(zip(("single", "multi"), FILLS[st["key"]]))
        upper = "multi" if st["multi"] >= st["single"] else "single"
        for arm in ("single", "multi"):
            y = st[arm]
            lo, hi = st[f"{arm}_ci"]
            ax.scatter([x], [y], s=DOT, color=fill[arm], zorder=4,
                       edgecolor=ring(fill[arm], theme), linewidth=EDGE_LW)
            ax.plot([x, x], [lo, hi], lw=CI_LW, color=t["ink"],
                    solid_capstyle="butt", zorder=5)
            for end in (lo, hi):
                ax.plot([x], [end], marker="_", ms=10, mew=CI_LW,
                        color=t["ink"], zorder=5)
            end, dy, va = (hi, 5, "bottom") if arm == upper else (lo, -5, "top")
            ax.annotate(f"{y:.1f}", xy=(x, end), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=FS_BODY, color=t["ink"])

    for i, st in enumerate(stats):
        col = t["ink2"] if st["sig"] else t["muted_text"]
        drop = model_block(ax, st["params"], [
            (head_label(st), FS_HEAD, "bold", t["ink"]),
            (f"\u0394 {st['delta']:+.1f} pp", FS_NOTE, "normal", col),
            (f"p {fmt_p_num(st['p_holm'], 1, st['floored'])}", FS_NOTE, "normal", col),
            (f"group {letters[i]}", FS_NOTE, "normal", t["ink2"])])

    ax.annotate("Total parameters, log scale", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -(drop + 12)), textcoords="offset points",
                ha="center", va="top", fontsize=FS_BODY, color=t["ink2"])

    for st in stats:
        y1, y2 = st["single"], st["multi"]
        xa, xb = x_offset(st["params"], gap), x_offset(st["params"], gap + 8)
        col = t["ink2"] if st["sig"] else t["muted"]
        ax.plot([xa, xb, xb, xa], [y1, y1, y2, y2], lw=1.6, color=col,
                solid_joinstyle="miter", zorder=2)
        mark = stars(st["p_holm"])
        top = max(st["single_ci"][1], st["multi_ci"][1])
        ax.annotate(mark or "n.s.", xy=(st["params"], top), xytext=(0, 26),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_STAR if mark else FS_NOTE,
                    color=t["ink"] if mark else t["muted_text"], zorder=5)

    pairs, names = [], []
    for st in stats:
        light, dark = FILLS[st["key"]]
        pairs.append(tuple(
            Line2D([], [], marker="o", ls="", ms=12, color=c,
                   markeredgecolor=ring(c, theme), markeredgewidth=EDGE_LW)
            for c in (light, dark)))
        names.append(st["label"])

    trend = [Line2D([], [], ls="--", lw=1.6, color=t["muted"],
                    label=f"Single call\n{totals['single']:+.1f} pts\n9B \u2192 2.8T"),
             Line2D([], [], ls="-", lw=1.6, color=t["muted"],
                    label=f"With manager\n{totals['multi']:+.1f} pts\n9B \u2192 2.8T")]

    fig.suptitle(wrap_title(TITLE), x=MARGINS["left"], ha="left", y=0.99,
                 va="top", fontsize=FS_TITLE, fontweight="bold", color=t["ink"],
                 linespacing=1.25)
    side_panel(fig, t, pairs, names, trend,
               trend_kw=dict(handletextpad=0.7, handlelength=2.2))
    if save:
        write_figure(fig, save)
    return fig


def main():
    stats = compute()

    hdr = (f"{'model':13s} {'single':>7s} {'multi':>7s} {'delta':>7s} {'95% CI':>16s} "
           f"{'p_perm':>10s} {'p_holm':>9s}  nonempty")
    print(hdr)
    print("-" * len(hdr))
    for st in stats:
        pre = "<" if st["floored"] else " "
        print(f"{st['label']:13s} {st['single']:7.1f} {st['multi']:7.1f} {st['delta']:+7.1f}"
              f"   [{st['delta_ci'][0]:+5.1f},{st['delta_ci'][1]:+6.1f}]"
              f" {pre}{st['p_perm']:9.2e} {st['p_holm']:9.2e}"
              f"  {st['ne_single']:.1f} -> {st['ne_multi']:.1f}")

    pmat, letters = tukey(stats)
    names = [st["label"] for st in stats]
    print("\nTukey HSD across models on the per-problem delta:")
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            diff = stats[i]["d"] - stats[j]["d"]
            _, p_paired, _ = perm_sign_p(diff, n_perm=50_000)
            print(f"  {names[i]:12s} vs {names[j]:12s}  diff {diff.mean():+6.1f} pp"
                  f"   Tukey p {pmat[i][j]:7.4f}   paired-permutation p {p_paired:7.4f}")
    print("  letters: " + ", ".join(f"{names[i]}={letters[i]}" for i in range(len(stats))))

    os.makedirs(PLOTS, exist_ok=True)
    for theme in ("light",):
        draw(stats, letters, theme,
             save=os.path.join(PLOTS, f"{slug(TITLE)}_{theme}.pdf"))


if __name__ == "__main__":
    main()
