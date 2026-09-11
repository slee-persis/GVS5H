#!/usr/bin/env python3
"""Manager-worker control flow, in plain English -- two diagrams of the same loop."""
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS = os.path.join(ROOT, "paper/plots")

DPI = 2820 / 11.0

THEMES = {
    "light": dict(
        surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        wire="#c3c2b7",
        fills=dict(lead="#948fbe", helper="#80c2ae", decision="#edd4b2", terminal="#f0eee7",
                   check="#a6c9e7"),
        edges=dict(lead="#4e4d8b", helper="#008168", decision="#ab8d5b", terminal="#a8a79c",
                   check="#4786af"),
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        wire="#4a4a46",
        fills=dict(lead="#6e6792", helper="#3b6d61", decision="#45392a", terminal="#2a2a25",
                   check="#3f4e62"),
        edges=dict(lead="#b3a8ea", helper="#69b5a3", decision="#8a765e", terminal="#66665d",
                   check="#7790ae"),
    ),
}

FS_TITLE, FS_HEAD, FS_SUB, FS_LABEL, FS_LEGEND, FS_NOTE = 18.0, 14.0, 14.0, 14.0, 14.0, 14.0

Y_STRETCH = 1.65

LS_HEAD, LS_SUB = 1.25, 1.30

DECISION_COLORS = dict(yes="#2f6fbf", no="#b5651d")

DECISION_STYLES = dict(yes="-", no=(0, (2, 1.5)))

SWATCH_W, SWATCH_H = 60, 42

# --- figure 1: multiagent.py as the current tree runs it, the paper's Figure 4 ----------

TITLE_CUR = "How the manager arm answers one question — multiagent.py, current tree"

BOXES_CUR = [
    ("terminal", 60, 110, 600, 175, "One question comes in", None),
    ("lead", 60, 215, 600, 380, "Lead agent: plan", "write a plan,\nlist 3–6 tasks"),
    ("helper", 60, 420, 600, 625, "Helper agent:\nbrainstorm",
     "suggest approaches\nonly (asked not to\nsolve it yet)"),
    ("lead", 60, 665, 600, 875, "Lead agent: manage",
     "tidy list, pick the\nnext task; cannot\nfinish on an empty\nworkspace"),
    ("decision", 750, 110, 1290, 355, "Did the last round\nfail the public\nsample tests?",
     "(a fail vetoes\nthe manager)"),
    ("decision", 750, 395, 1290, 500, "Is the work\nfinished?", None),
    ("decision", 750, 540, 1290, 745, "Is an answer\nalready saved?",
     "(check the\nworkspace)"),
    ("terminal", 750, 785, 1290, 905, "Skip", "the write-up"),
    ("decision", 1440, 110, 1980, 355, "Another task left,\nand fewer than 10\nrounds used?",
     "(none named → first\nunfinished task)"),
    ("decision", 1440, 395, 1980, 515, "Is it the same task", "we just handed out?"),
    ("helper", 1440, 555, 1980, 760, "Helper agent:\ndo the task",
     "updates the answer,\nrewrites the notes"),
    ("helper", 1440, 800, 1980, 920, "Helper agent writes", "the final answer"),
    ("terminal", 1440, 960, 1980, 1125, "Hand back the solution",
     "and answer files\nfor grading"),
    ("decision", 2130, 110, 2670, 230, "Was it cut off at", "the token limit?"),
    ("helper", 2130, 270, 2670, 475, "Helper agent:\nsummarize",
     "the cut-off\nattempt"),
    ("decision", 2130, 515, 2670, 720, "Did this round write\nfresh code?",
     "(and does the problem\nship stdin tests?)"),
    ("check", 2130, 760, 2670, 965, "Run solution.py\non the samples",
     "verdict goes to\nthe manager as\nground truth"),
]

WIRES_CUR = [
    ([(330, 175), (330, 215)], None),
    ([(330, 380), (330, 420)], None),
    ([(330, 625), (330, 665)], None),
    ([(600, 770), (680, 770), (680, 232), (750, 232)], None),
    ([(1020, 355), (1020, 395)], "no"),
    ([(1290, 232), (1400, 232), (1400, 180), (1440, 180)], "yes"),
    ([(1020, 500), (1020, 540)], "yes"),
    ([(1290, 447), (1360, 447), (1360, 310), (1440, 310)], "no"),
    ([(1020, 745), (1020, 785)], "yes"),
    ([(1290, 642), (1310, 642), (1310, 910), (1440, 910)], "no"),
    ([(1020, 905), (1020, 1042), (1440, 1042)], None),
    ([(1710, 355), (1710, 395)], "yes"),
    ([(1440, 335), (1400, 335), (1400, 890), (1440, 890)], "no"),
    ([(1440, 455), (1420, 455), (1420, 830), (1440, 830)], "yes"),
    ([(1710, 515), (1710, 555)], "no"),
    ([(1980, 657), (2060, 657), (2060, 170), (2130, 170)], None),
    ([(2400, 230), (2400, 270)], "yes"),
    ([(2130, 200), (2075, 200), (2075, 617), (2130, 617)], "no"),
    ([(2400, 475), (2400, 495), (2550, 495), (2550, 515)], None),
    ([(2400, 720), (2400, 760)], "yes"),
    ([(2670, 617), (2740, 617), (2740, 1145)], "no"),
    ([(2670, 862), (2740, 862), (2740, 1145), (330, 1145), (330, 875)], None),

]

LABELS_CUR = []

LEGEND_CUR = [("lead", "lead agent", 60), ("helper", "helper agent", 315),
              ("decision", "decision", 600), ("terminal", "start / end", 820),
              ("check", "sample-test run (no model call)", 1070),
              ("yes", "decision: yes", 1620), ("no", "decision: no", 1910)]


# --- figure 2: multiagent.py @ 46710a5, the ORIGINAL scaffold (§2.3) --------------------

TITLE_46 = "How the manager arm answers one question — multiagent.py @ 46710a5"

BOXES_46 = [
    ("terminal", 324, 106, 545, 155, "One question comes in", None),
    ("lead", 318, 196, 552, 264, "Lead agent: plan", "write a plan, list 3–6 tasks"),
    ("helper", 298, 304, 571, 393, "Helper agent: brainstorm",
     "suggest approaches only\n(asked not to solve it yet)"),
    ("lead", 298, 427, 571, 522, "Lead agent: manage",
     "tidy list, pick the next task\ncannot finish on an empty workspace"),
    ("decision", 331, 565, 538, 621, "Is the work finished?", None),
    ("decision", 93, 661, 327, 736, "Is an answer already saved?", "(check the workspace)"),
    ("decision", 615, 654, 888, 743, "Another task left, and fewer\nthan 4 rounds used?",
     "(none named → first unfinished task)"),
    ("terminal", 80, 803, 221, 859, "Skip", "the write-up"),
    ("decision", 621, 793, 882, 868, "Is it the same task", "we just handed out?"),
    ("helper", 258, 925, 505, 1000, "Helper agent writes", "the final answer"),
    ("helper", 615, 925, 888, 1000, "Helper agent: do the task", "updates answer + what's next"),
    ("terminal", 245, 1064, 624, 1126, "Hand back the solution", "and answer files for grading"),
]

WIRES_46 = [
    ([(434.5, 155), (434.5, 196)], None),
    ([(434.5, 264), (434.5, 304)], None),
    ([(434.5, 393), (434.5, 427)], None),
    ([(434.5, 522), (434.5, 565)], None),
    ([(331, 593), (210, 593), (210, 661)], None),
    ([(538, 593), (751, 593), (751, 654)], None),
    ([(150, 736), (150, 803)], None),
    ([(290, 736), (290, 925)], None),
    ([(751, 743), (751, 793)], None),
    ([(751, 868), (751, 925)], None),
    ([(615, 698), (368, 698), (368, 925)], None),
    ([(621, 830), (447, 830), (447, 925)], None),
    ([(888, 962), (924, 962), (924, 474), (571, 474)], None),
    ([(150, 859), (150, 1095), (245, 1095)], None),
    ([(381.5, 1000), (381.5, 1064)], None),
]

LABELS_46 = [
    ("yes", 270, 575), ("no", 645, 575),
    ("back to the manager", 752, 451),
    ("no — out of rounds", 491, 678),
    ("yes", 128, 768), ("no", 323, 758),
    ("yes", 778, 767), ("no", 778, 895),
    ("yes (nothing changed)", 534, 810),
]

LEGEND_46 = [("lead", "lead agent", 80), ("helper", "helper agent", 311),
             ("decision", "decision", 542), ("terminal", "start / end", 773)]

NOTES_46 = [
    "The manager owns the loop; the workspace files are the only state a role sees, "
    "and no task carries a result field.",
    "•  “Is an answer already saved?” is always yes on code problems — the manager "
    "cannot report finished with an empty",
    "    solution.py, so the “no” branch is reachable only for math "
    "(an answer.md holding no ANSWER: line).",
    "•  4 rounds is the MULTIAGENT_MAX_ITERS default at this commit; "
    "the uncommitted tree raises it to 10.",
    "•  A worker's own “solved” claim never ends the loop — only the manager's verdict does.",
]
NOTE_DY = 21.5


FIG_CURRENT = dict(title=TITLE_CUR, size=(2820, 1250), y_pad=0, boxes=BOXES_CUR,
                   wires=WIRES_CUR, labels=LABELS_CUR, legend=LEGEND_CUR, legend_y=1180)

FIG_46710A5 = dict(title=TITLE_46, size=(1015, 1354), boxes=BOXES_46, wires=WIRES_46,
                   labels=LABELS_46, legend=LEGEND_46, legend_y=1156,
                   notes=NOTES_46, note_y0=1218)

LEGEND_PAD = 14
LEGEND_GAP = 52
LEGEND_ROW = 66


def layout_legend(entries, width, y0):
    fp = FontProperties(family=["Liberation Sans"])
    x0 = entries[0][2]
    avail = width - 2 * x0
    widths = [SWATCH_W + LEGEND_PAD
              + TextPath((0, 0), text, size=FS_LEGEND,
                         prop=fp).get_extents().width * DPI / 72
              for _, text, _ in entries]

    def rows_of(n):
        per = math.ceil(len(entries) / n)
        chunks = [list(range(i, min(i + per, len(entries))))
                  for i in range(0, len(entries), per)]
        for c in chunks:
            if sum(widths[i] for i in c) + LEGEND_GAP * (len(c) - 1) > avail:
                return None
        return chunks

    chunks = next((r for n in range(1, len(entries) + 1) if (r := rows_of(n))), None)
    if chunks is None:
        chunks = [[i] for i in range(len(entries))]

    out = []
    for row, chunk in enumerate(chunks):
        x = x0
        for i in chunk:
            out.append((entries[i][0], entries[i][1], x, y0 + row * LEGEND_ROW))
            x += widths[i] + LEGEND_GAP
    return out


def stretch_y(figure, k=Y_STRETCH):
    def sy(v):
        return v * k

    W, H = figure["size"]
    out = dict(figure, size=(W, round(H * k)), y_pad=round(figure.get("y_pad", 0) * k))
    out["boxes"] = [(kind, x0, sy(y0), x1, sy(y1), head, sub)
                    for kind, x0, y0, x1, y1, head, sub in figure["boxes"]]
    out["wires"] = [([(x, sy(y)) for x, y in pts], branch)
                    for pts, branch in figure["wires"]]
    out["labels"] = [(text, x, sy(y)) for text, x, y in figure["labels"]]
    out["legend_y"] = sy(figure["legend_y"])
    return out


FIGURES = [stretch_y(FIG_CURRENT)]


def slug(text):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9-]+", "_",
                                     text.lower().replace("—", " "))).strip("_")


def draw(figure, theme="light", save=None):
    t = THEMES[theme]
    W, H = figure["size"]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "Helvetica"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.size": 10,
        "mathtext.default": "regular", "mathtext.fontset": "dejavusans",
    })
    y_pad = figure.get("y_pad", 0)
    fig = plt.figure(figsize=(W / DPI, (H + y_pad) / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, -y_pad))
    ax.axis("off")
    ax.set_facecolor(t["surface"])

    ax.text(W / 2, 72 * Y_STRETCH, figure["title"], ha="center", va="center",
            fontsize=FS_TITLE, fontweight="bold", color=t["ink"])

    for kind, x0, y0, x1, y1, head, sub in figure["boxes"]:
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=6",
            facecolor=t["fills"][kind], edgecolor=t["edges"][kind], linewidth=1.1, zorder=2))
        if sub is None:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, head, ha="center", va="center",
                    fontsize=FS_HEAD, color=t["ink"], zorder=3, linespacing=LS_HEAD)
        else:
            nh, ns = head.count("\n") + 1, sub.count("\n") + 1
            head_h = nh * FS_HEAD * LS_HEAD * DPI / 72
            sub_h = ns * FS_SUB * LS_SUB * DPI / 72
            block_gap = 16
            total = head_h + block_gap + sub_h
            top = (y0 + y1) / 2 - total / 2
            head_y = top + head_h / 2
            sub_y = top + head_h + block_gap + sub_h / 2
            ax.text((x0 + x1) / 2, head_y, head, ha="center", va="center",
                    fontsize=FS_HEAD, color=t["ink"], zorder=3, linespacing=LS_HEAD)
            ax.text((x0 + x1) / 2, sub_y, sub, ha="center", va="center",
                    fontsize=FS_SUB, color=t["ink"], zorder=3, linespacing=LS_SUB)

    for pts, branch in figure["wires"]:
        color = DECISION_COLORS.get(branch, t["ink"])
        style = DECISION_STYLES.get(branch, "-")
        lw = 2.4 if branch else 1.2
        if len(pts) > 2:
            ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                    color=color, lw=lw, ls=style, solid_capstyle="butt",
                    solid_joinstyle="miter", dash_capstyle="butt", zorder=1)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=style, shrinkA=0, shrinkB=0,
                                    mutation_scale=9), zorder=1)

    for text, x, y in figure["labels"]:
        ax.text(x, y, text, ha="center", va="center", fontsize=FS_LABEL, color=t["ink"])

    for kind, text, x, row_y in layout_legend(figure["legend"], W, figure["legend_y"]):
        if kind in DECISION_COLORS:
            y = row_y + SWATCH_H / 2
            ax.plot([x, x + SWATCH_W], [y, y], color=DECISION_COLORS[kind], lw=2.4,
                    ls=DECISION_STYLES[kind], dash_capstyle="butt",
                    solid_capstyle="butt")
        else:
            ax.add_patch(FancyBboxPatch(
                (x, row_y), SWATCH_W, SWATCH_H,
                boxstyle="round,pad=0,rounding_size=5",
                facecolor=t["fills"][kind], edgecolor=t["edges"][kind], linewidth=1.1))
        ax.text(x + SWATCH_W + LEGEND_PAD, row_y + SWATCH_H / 2, text,
                ha="left", va="center", fontsize=FS_LEGEND, color=t["ink"])

    for i, note in enumerate(figure.get("notes", ())):
        ax.text(78, figure["note_y0"] + NOTE_DY * i, note, ha="left", va="center",
                fontsize=FS_NOTE, color=t["ink"])

    if save:
        stem = os.path.splitext(save)[0]
        fig.savefig(stem + ".pdf")
        print("wrote", stem + ".pdf")
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for figure in FIGURES:
        for theme in ("light",):
            draw(figure, theme,
                 save=os.path.join(PLOTS, f"{slug(figure['title'])}_{theme}.pdf"))


if __name__ == "__main__":
    main()
