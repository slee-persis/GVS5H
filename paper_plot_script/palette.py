"""The one model palette. Every chart imports this; no script defines its own.

There used to be five FILLS dicts across these scripts, two of them hand-copied
under a comment calling itself "the shared model palette". They had drifted:
#a8cdf0 meant GPT-5.6-Luna in the pinned-backend figures and MiniMax-M3 in the
scale figures, and #a8d878 meant GPT-5.6-Terra in one and Kimi-K3 in the other.
A reader flipping between Figure 1 and Figure 4 saw one colour name two models.

Three channels, and the model rides two of them:

    fill lightness -> ARM      light = single call, dark = with manager
    hue            -> MODEL    for colour readers
    hatch / marker -> MODEL    again, so identity survives grayscale and CVD

The old palette spent lightness on BOTH arm and model, giving each model's dark
fill its own level. That cannot work: separating 4 models x 2 arms needs 8
distinguishable greys, at which point "light = single, dark = manager" stops
reading as a pair relationship -- and the captions all state that convention.
The worst pairs came out 0.2-0.7 L* apart, i.e. the same grey. So lightness now
carries the arm alone, at ONE level per arm 47 L* apart, and the texture channel
carries the model. That is why the hatch is load-bearing rather than decorative:
deuteranopia dE between manager fills bottoms out at 10.4 (Qwen3.5-9B vs
MiniMax-M3, Figures 4 and 6), which is only legal WITH a secondary encoding.
Drop the hatch and that pair stops being distinguishable -- see main() below,
which measures every gate.

Hues are the dataviz reference palette's categorical order, plus a cyan for
Opus-5, stepped to a fixed L* per arm with an absolute Lab chroma cap so nothing
prints neon. Qwen3.5-9B and Opus-5 share texture slot 4: they never appear in
the same figure, and they carry different hues anyway.

    uv run --with matplotlib python paper_plot_script/palette.py            # gates
    uv run --with matplotlib python paper_plot_script/palette.py --swatches # sheet
"""
import itertools
import math
import os

# key -> (single call, with manager). Same shape the old per-script FILLS had,
# so importing this is a one-line change at each call site.
FILLS = {
    "q38":   ("#d1dcff", "#005fb6"),   # Qwen3.8-27B     blue
    "terra": ("#ffd3c1", "#a23f17"),   # GPT-5.6-Terra   orange
    "luna":  ("#a7ebc9", "#006c48"),   # GPT-5.6-Luna    aqua
    "fable": ("#ffd2cc", "#ad3132"),   # Claude Fable 5  red
    "q35":   ("#ffd0df", "#ad2666"),   # Qwen3.6-35B     magenta
    "mm3":   ("#c1e7b4", "#1b6d12"),   # MiniMax-M3      green
    "kimi":  ("#e4d6ff", "#624fae"),   # Kimi-K3         violet
    "q9":    ("#fed6a6", "#815600"),   # Qwen3.5-9B      yellow
    "opus":  ("#8aebfe", "#006877"),   # Opus-5          cyan
}

# The texture slot. Fixed per model, so a model's hatch is the same in every
# figure it appears in. Only 4 slots are needed: no figure shows more than four
# models. q9 and opus share slot 4 (disjoint figures).
SLOT = {"q38": 1, "terra": 2, "luna": 3, "fable": 4,
        "q35": 1, "mm3": 2, "kimi": 3, "q9": 4, "opus": 4}

# Slot 1 is deliberately untextured: one clean fill per figure keeps the others
# reading as texture rather than as noise. Density is "medium" -- at the sparse
# setting a 0.38-high bar shows only two or three strokes, at dense the
# cross-hatch fills in and stops being distinguishable from solid.
HATCH = {1: "", 2: "//", 3: "..", 4: "xx"}
MARKER = {1: "o", 2: "s", 3: "^", 4: "D"}      # scatters: hatch is invisible on a 6.9pt disc

HATCH_LW = 0.40     # printed points; multiply by the script's PAGE_SCALE
SURFACE = "#ffffff"  # hatch ink on a dark fill

LABELS = {"q38": "Qwen3.8-27B", "terra": "GPT-5.6-Terra", "luna": "GPT-5.6-Luna",
          "fable": "Claude Fable 5", "q35": "Qwen3.6-35B", "mm3": "MiniMax-M3",
          "kimi": "Kimi-K3", "q9": "Qwen3.5-9B", "opus": "Opus-5"}

# The model sets that actually share a figure. The gates below are checked
# within each set, not across all nine -- two models that never co-occur do not
# have to be told apart.
SETS = {
    "pinned  (Fig 1, 2, 3)": ["q38", "terra", "luna", "fable"],
    "scale off (Fig 4, 6)":  ["q9", "q35", "mm3", "kimi"],
    "scale on  (Fig 5)":     ["q35", "mm3", "kimi", "opus"],
}


def bar_kw(key, arm, surface=SURFACE):
    """Fill + texture for one bar. `arm` is "single" or "manager".

    The hatch inverts on the dark fill -- surface-coloured lines on the manager
    bar, the model's own dark on the pale single bar. Without the inversion the
    hatch is drawn in the fill's own colour and the manager bars all come out
    flat, which is the failure this whole channel exists to prevent.

    A Patch's edgecolor sets BOTH its outline and its hatch ink, so a hatched bar
    cannot also carry ring()'s hairline -- edgecolor is returned only when there
    IS a hatch, leaving the caller's own ring in place for slot 1. The hatch
    STROKE width is rcParams["hatch.linewidth"] (set in apply_theme), not the
    patch's linewidth, which stays the caller's edge width.
    """
    light, dark = FILLS[key]
    h = HATCH[SLOT[key]]
    kw = {"color": dark if arm == "manager" else light}
    if h:
        kw["hatch"] = h
        kw["edgecolor"] = surface if arm == "manager" else dark
    return kw


def marker_kw(key, arm):
    """Fill + shape for one scatter point."""
    light, dark = FILLS[key]
    return {"marker": MARKER[SLOT[key]],
            "color": dark if arm == "manager" else light,
            "edgecolor": dark}


# --------------------------------------------------------------- colour maths
# Kept here so the palette verifies itself with no other dependency.
def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hexs(t):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in t)


def _unlin(c):
    c = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return c * 255


def luminance(h):
    r, g, b = [_lin(v) for v in _rgb(h)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    la, lb = max(la, lb), min(la, lb)
    return (la + 0.05) / (lb + 0.05)


def lab(h):
    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (903.3 * t + 16) / 116
    r, g, b = [_lin(v) for v in _rgb(h)]
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = f(X), f(Y), f(Z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e(a, b):
    return math.dist(lab(a), lab(b))


def deuter(h):
    """Brettel-style deuteranope simulation."""
    r, g, b = [_lin(v) * 255 for v in _rgb(h)]
    L = 17.8824 * r + 43.5161 * g + 4.11935 * b
    S = 0.0299566 * r + 0.184309 * g + 1.46709 * b
    M = 0.494207 * L + 1.24827 * S
    R = 0.080944 * L - 0.130504 * M + 0.116721 * S
    G = -0.0102485 * L + 0.0540194 * M - 0.113615 * S
    B = -0.000365294 * L - 0.00412163 * M + 0.693513 * S
    return _hexs((_unlin(max(0, min(1, R / 255))), _unlin(max(0, min(1, G / 255))),
                  _unlin(max(0, min(1, B / 255)))))


ARM_MIN = 3.0       # single vs manager fill of one model; not text, so AA does not bind
DEUTER_MIN = 8.0    # manager fills within a figure; the WITH-secondary-encoding floor
ARM_GULF_MIN = 30.0  # L* between the pale band and the dark band


def main():
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)
        return "ok  " if ok else "FAIL"

    print("arm contrast per model (need %.1f:1)" % ARM_MIN)
    for k, (l, d) in FILLS.items():
        r = contrast(l, d)
        print(f"  {check(r >= ARM_MIN, f'{k} arm pair {r:.2f}:1')} "
              f"{LABELS[k]:16s} {l} -> {d}  {r:.2f}:1")

    print(f"\nmanager fills under deuteranopia, within a figure (need dE >= {DEUTER_MIN:.0f};"
          f" hatch is the secondary encoding that makes this floor legal)")
    for name, keys in SETS.items():
        for a, b in itertools.combinations(keys, 2):
            e = delta_e(deuter(FILLS[a][1]), deuter(FILLS[b][1]))
            print(f"  {check(e >= DEUTER_MIN, f'{name}: {a} vs {b} dE {e:.1f}')} "
                  f"{name:22s} {LABELS[a]:15s} vs {LABELS[b]:15s} dE {e:5.1f}")

    print(f"\ngrayscale arm separation (need {ARM_GULF_MIN:.0f} L*)")
    pale = min(lab(v[0])[0] for v in FILLS.values())
    dark = max(lab(v[1])[0] for v in FILLS.values())
    print(f"  {check(pale - dark >= ARM_GULF_MIN, f'arm gulf {pale - dark:.0f} L*')} "
          f"palest single L* {pale:.0f} vs darkest manager L* {dark:.0f} -> gulf {pale - dark:.0f}")

    print("\ntexture distinct within a figure")
    for name, keys in SETS.items():
        hs = [HATCH[SLOT[k]] for k in keys]
        ms = [MARKER[SLOT[k]] for k in keys]
        print(f"  {check(len(set(hs)) == len(hs) and len(set(ms)) == len(ms), f'{name} texture clash')} "
              f"{name:22s} hatch {hs}  marker {ms}")

    print("\none hue per model across the paper")
    seen = {}
    for k, (l, d) in FILLS.items():
        for c in (l, d):
            seen.setdefault(c, []).append(k)
    dupes = {c: ks for c, ks in seen.items() if len(ks) > 1}
    print(f"  {check(not dupes, f'shared hex: {dupes}')} "
          f"{len(FILLS)} models, {len(seen)} distinct fills")

    print("\n%d FAILURES" % len(fails))
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


def swatches(path=None):
    """Render the palette in colour and grayscale, so a change is inspectable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "palette_swatches.png")
    matplotlib.rcParams["hatch.linewidth"] = HATCH_LW * 2

    def grey(c):
        y = luminance(c)
        v = 1.055 * (y ** (1 / 2.4)) - 0.055 if y > 0.0031308 else 12.92 * y
        return (v, v, v)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, gray in zip(axes, (False, True)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 15)
        ax.axis("off")
        ax.set_title("colour" if not gray else "grayscale (a b/w printer)",
                     fontsize=11, fontweight="bold")
        y = 14.0
        for name, keys in SETS.items():
            ax.text(0, y, name, fontsize=8.5, fontweight="bold", color="#52514e")
            y -= 0.8
            for k in keys:
                light, dark = FILLS[k]
                for i, (c, arm) in enumerate(((light, "single"), (dark, "manager"))):
                    face = grey(c) if gray else c
                    edge = (grey(SURFACE) if gray else SURFACE) if arm == "manager" else \
                           (grey(dark) if gray else dark)
                    ax.add_patch(Rectangle((3.6 + i * 1.5, y - 0.34), 1.4, 0.62,
                                           facecolor=face, hatch=HATCH[SLOT[k]],
                                           edgecolor=edge, linewidth=0.8))
                ax.text(3.5, y - 0.03, LABELS[k], fontsize=8, ha="right", va="center")
                y -= 0.8
            y -= 0.4
        ax.text(3.6, y + 0.4, "single call        with manager", fontsize=7.5,
                color="#898781")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    import sys
    if "--swatches" in sys.argv:
        swatches()
    else:
        sys.exit(main())
