"""The one model palette. Every chart imports this; no script defines its own."""
import itertools
import math
import os

FILLS = {
    "q38":   ("#d1dcff", "#005fb6"),
    "terra": ("#ffd3c1", "#a23f17"),
    "luna":  ("#a7ebc9", "#006c48"),
    "fable": ("#ffd2cc", "#ad3132"),
    "q35":   ("#ffd0df", "#ad2666"),
    "mm3":   ("#c1e7b4", "#1b6d12"),
    "kimi":  ("#e4d6ff", "#624fae"),
    "q9":    ("#fed6a6", "#815600"),
    "opus":  ("#8aebfe", "#006877"),
}

SLOT = {"q38": 1, "terra": 2, "luna": 3, "fable": 4,
        "q35": 1, "mm3": 2, "kimi": 3, "q9": 4, "opus": 4}

HATCH = {1: "", 2: "//", 3: "..", 4: "xx"}
MARKER = {1: "o", 2: "s", 3: "^", 4: "D"}

HATCH_LW = 0.40
SURFACE = "#ffffff"

LABELS = {"q38": "Qwen3.8-27B", "terra": "GPT-5.6-Terra", "luna": "GPT-5.6-Luna",
          "fable": "Claude Fable 5", "q35": "Qwen3.6-35B", "mm3": "MiniMax-M3",
          "kimi": "Kimi-K3", "q9": "Qwen3.5-9B", "opus": "Opus-5"}

SETS = {
    "pinned  (Fig 1, 2, 3)": ["q38", "terra", "luna", "fable"],
    "scale off (Fig 4, 6)":  ["q9", "q35", "mm3", "kimi"],
    "scale on  (Fig 5)":     ["q35", "mm3", "kimi", "opus"],
}


def bar_kw(key, arm, surface=SURFACE):
    light, dark = FILLS[key]
    h = HATCH[SLOT[key]]
    kw = {"color": dark if arm == "manager" else light}
    if h:
        kw["hatch"] = h
        kw["edgecolor"] = surface if arm == "manager" else dark
    return kw


def marker_kw(key, arm):
    light, dark = FILLS[key]
    return {"marker": MARKER[SLOT[key]],
            "color": dark if arm == "manager" else light,
            "edgecolor": dark}


# --------------------------------------------------------------- colour maths
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
    r, g, b = [_lin(v) * 255 for v in _rgb(h)]
    L = 17.8824 * r + 43.5161 * g + 4.11935 * b
    S = 0.0299566 * r + 0.184309 * g + 1.46709 * b
    M = 0.494207 * L + 1.24827 * S
    R = 0.080944 * L - 0.130504 * M + 0.116721 * S
    G = -0.0102485 * L + 0.0540194 * M - 0.113615 * S
    B = -0.000365294 * L - 0.00412163 * M + 0.693513 * S
    return _hexs((_unlin(max(0, min(1, R / 255))), _unlin(max(0, min(1, G / 255))),
                  _unlin(max(0, min(1, B / 255)))))


ARM_MIN = 3.0
DEUTER_MIN = 8.0
ARM_GULF_MIN = 30.0


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
