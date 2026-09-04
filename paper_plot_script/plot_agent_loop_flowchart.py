#!/usr/bin/env python3
"""Manager-worker control flow, in plain English -- two diagrams of the same loop.

FIG_CURRENT is the paper's Figure 4: the loop as multiagent.py runs it today, i.e. the v2
scaffold that S3.1 describes and that produced the pinned-backend arms of S2.1. The round
budget is 10; after a round that wrote fresh code, solution.py is run against the problem's
public stdin samples and the verdict is handed to the manager as ground truth (and vetoes a
"done" verdict); and a worker cut off at the token limit gets its partial attempt digested
by an extra call before the manager sees the round. Its layout is its own grid, not the
source PNG's.

FIG_46710A5 documents the ORIGINAL scaffold -- the version behind the OpenRouter-served
results in S2.3 -- and is no longer the figure the paper includes. It is a reconstruction of
`paper/agent_loop_flowchart_plain_english.png`, whose generating script was lost. Box
positions, fills, borders and font sizes were measured back off that PNG, so the output
matches it closely; it is a redraw, not a pixel copy. It documents multiagent.py @
46710a5, which is what its title records: MULTIAGENT_MAX_ITERS was 4 there, and the
sample-test verifier does not appear at all. Keep it -- it is the only drawing of the
scaffold that produced S2.3, and the paper's S3 table of v2-vs-original differences is
read off the pair.

    uv run --with matplotlib python paper_plot_script/plot_agent_loop_flowchart.py

Writes paper/plots/<title-slug>_light.pdf for FIG_CURRENT, the paper's figure.
"""
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
ROOT = os.path.dirname(HERE)             # repo root, one level up from paper_plot_script/
PLOTS = os.path.join(ROOT, "paper/plots")      # every chart script writes here

# Each figure carries its own canvas, a pixel grid with y running downward. For the
# 46710a5 figure that grid is the source PNG's, so every coordinate in it is a
# measurement off that image rather than a guess.
#
# DPI ties pixel coordinates to the same "authored at 2x, prints at 1x" convention
# plot_16k_reason_off_5_pass.py uses (see its PAGE_SCALE / DESIGN_W_IN notes): the
# CURRENT-tree canvas is 2820px wide, and 2820/DPI must equal 2 x the design width for
# FS_* below to print at exactly their stated point value. That width is this document's
# \linewidth, 5.5in, so 2820/11.0 = 256.4. It used to be Nature's 180mm, which is the
# bug plot_16k_reason_off_5_pass.py's DESIGN_W_IN comment describes.
#
# Only type scales with DPI. A box's PRINTED size is box_px x 5.5/2820, independent of
# it -- which is why raising the band needed Y_STRETCH below as well.
DPI = 2820 / 11.0

# The hues are the ones sampled from the source PNG; the LIGHTNESS is not. The five
# fills used to sit inside 3.0 L* of each other (0.28-1.50 between neighbours), which is
# a hue-only encoding: printed in black and white, or read by anyone who cannot separate
# those hues, all five collapsed to the same grey and the legend stopped meaning
# anything. They now sit on an even ramp -- lightest for the harness steps, darkest for
# the two kinds of model call -- so lightness alone names a box. Each hue angle is held,
# so the colour version reads as it did.
#
# The ramp is even in Rec.601 LUMA, not in L*: that is the transform pdftoppm, a
# photocopier and a mono printer all apply, and an L*-even ramp came out lopsided
# through it (22/25/28/11 levels between neighbours instead of ~22 each). The five are
# 22 apart in the light theme and 17 in the dark, against the 1-2 they had.
#
# Every fill clears WCAG 2.1 AA on its own text: 6.5:1 at the dark end of the light
# theme, 5.2:1 at the light end of the dark one. `check` is the one fill the source PNG
# has no value for -- it marks the sample-test run, the only step in either diagram that
# is not a model call. `terminal` is held near-neutral (chroma 4) because start/end is
# not a category the reader has to name by hue.
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

# Stated as the point size each should PRINT at, doubled for the authored canvas -- the
# same 7-9pt band and the same convention plot_16k_reason_off_5_pass.py's FS_* comment
# explains in full. HEAD and SUB sit on the 7pt floor: several boxes carry a 3-line
# heading or a 3-line subtext, and every point above the floor costs the canvas height
# that Y_STRETCH below has to pay for.
FS_TITLE, FS_HEAD, FS_SUB, FS_LABEL, FS_LEGEND, FS_NOTE = 18.0, 14.0, 14.0, 14.0, 14.0, 14.0

# The boxes were measured for the old 5.5pt-printed text and hold 1.45/1.6-spaced lines
# with no room over. Type at 7pt is 1.64x taller in a box whose PRINTED size does not
# move with DPI, so every one of the 17 overflowed. Stretching the y axis of the authored
# grid -- box tops and bottoms, wire waypoints, labels, the legend row and the canvas
# height, all by the same factor -- makes the diagram taller on the page and leaves the
# column structure, the routing and the horizontal measurements exactly as authored.
# 1.65 is the smallest factor that clears the tallest box (392px of text in 245px).
Y_STRETCH = 1.65

# Leading inside a box. The old 1.45/1.6 was set when the type was 5.5pt and the air
# between lines was what made it readable; at 7pt the glyphs carry that themselves, and
# the three boxes with the most text only fit under Y_STRETCH once the leading came in.
LS_HEAD, LS_SUB = 1.25, 1.30

# CUR's decision branches are colour-coded rather than tagged "yes"/"no" at all 14 of
# them (see the WIRES_CUR comment). Checked against research-figure-guide.nature.com's
# accessible-palette requirement the same way the model colours in
# plot_16k_reason_off_5_pass.py are: CIE76 dE 101 in normal vision, 112 simulated for
# deuteranopia, so the two read
# apart under the confusion a colourblind reader would face. The legend spells out which
# is which in text, so the mapping is never colour-only.
DECISION_COLORS = dict(yes="#2f6fbf", no="#b5651d")

# ...and a line style as well as a colour. The two hues are 4.3 L* apart, so in black
# and white -- or in the legend's two short swatches, where there is little ink to judge
# a hue from -- they were the same grey line. Solid for yes, dashed for no is a second,
# fully redundant channel: no reader has to resolve the colour to follow a branch.
# The dash is short on purpose. matplotlib scales a dash spec by the line width, so at
# lw 2.4 a (5, 3) pattern is a 12pt dash -- longer than the 16.9pt legend swatch, which
# drew "decision: no" as one stroke indistinguishable from the solid "yes". (2, 1.5) is
# 4.8pt on, 3.6pt off: two full cycles inside the swatch, and still plainly dashed on a
# wire that runs the width of the figure.
DECISION_STYLES = dict(yes="-", no=(0, (2, 1.5)))

SWATCH_W, SWATCH_H = 60, 42             # legend swatch, shared by both figures. Sized
                                        # against FS_LEGEND rather than left at the old
                                        # 29x19: a swatch is how the reader reads a fill
                                        # off the key, and the lightness ramp above needs
                                        # enough area to be judged against its neighbour.

# --- figure 1: multiagent.py as the current tree runs it, the paper's Figure 4 ----------
# Same loop, wider canvas: the sample-test gate and the cut-off digest are two more steps
# in the worker column, and the manager's verdict is now checked against the gate first.
# Ordering note: in code the manager answers done/continue and the gate then overrides a
# "done" that fails the samples. A failing gate forces continue either way, so asking the
# gate FIRST -- as drawn -- is the same control flow with one fewer wire.

TITLE_CUR = "How the manager arm answers one question — multiagent.py, current tree"

# Four columns left to right (prep chain, the "finished" branch, the "keep going" branch,
# and the cut-off/verdict cluster) instead of one tall stack: research-figure-guide.
# nature.com caps a main figure at 170mm deep, and the original single-column layout
# (1200x1600px, aspect 1.33) printed 31mm over that with its smallest labels under the
# 5pt floor. Every heading/subtext line is kept under ~20 characters (rewrapped from the
# original where needed) so a 520px-wide box holds it at this figure's font size -- see
# the DPI note above for why that size is fixed, not a free choice.
#
# (kind, x0, y0, x1, y1, heading, subtext)
#
# Every column starts at the same y (110, clearing the title above it) instead of at
# whatever row its predecessor happened to occupy -- an earlier version row-aligned each
# column with the box that fed it (gate under manage, rounds under finished, cutoff under
# dotask), which left the whole upper-right quadrant empty: gate alone doesn't reach the
# canvas top until manage, four boxes into column 1, does. Packing every column tight
# from the top instead cuts the deepest column from 2140px to 1015px and turns that dead
# quadrant into content; the wires that used to be short horizontal hops because their
# two boxes shared a row are now elbows with a vertical jog, which costs some visual
# complexity for the space back.
BOXES_CUR = [
    # column 1: the fixed prep chain, top to bottom
    ("terminal", 60, 110, 600, 175, "One question comes in", None),
    ("lead", 60, 215, 600, 380, "Lead agent: plan", "write a plan,\nlist 3–6 tasks"),
    ("helper", 60, 420, 600, 625, "Helper agent:\nbrainstorm",
     "suggest approaches\nonly (asked not to\nsolve it yet)"),
    ("lead", 60, 665, 600, 875, "Lead agent: manage",
     "tidy list, pick the\nnext task; cannot\nfinish on an empty\nworkspace"),
    # column 2: the sample gate and the "finished" branch
    ("decision", 750, 110, 1290, 355, "Did the last round\nfail the public\nsample tests?",
     "(a fail vetoes\nthe manager)"),
    ("decision", 750, 395, 1290, 500, "Is the work\nfinished?", None),
    ("decision", 750, 540, 1290, 745, "Is an answer\nalready saved?",
     "(check the\nworkspace)"),
    ("terminal", 750, 785, 1290, 905, "Skip", "the write-up"),
    # column 3: the "keep going" branch, then the two convergence boxes reusing its
    # lower half once the branch itself has emptied out
    ("decision", 1440, 110, 1980, 355, "Another task left,\nand fewer than 10\nrounds used?",
     "(none named → first\nunfinished task)"),
    ("decision", 1440, 395, 1980, 515, "Is it the same task", "we just handed out?"),
    ("helper", 1440, 555, 1980, 760, "Helper agent:\ndo the task",
     "updates the answer,\nrewrites the notes"),
    ("helper", 1440, 800, 1980, 920, "Helper agent writes", "the final answer"),
    ("terminal", 1440, 960, 1980, 1125, "Hand back the solution",
     "and answer files\nfor grading"),
    # column 4: cut-off check, its summarize detour, and the verdict run. summarize sits
    # in its own row rather than sharing the cut-off/fresh-code channel, so the "no"
    # branch can drop straight from one to the other without crossing its box -- the
    # same reason the original parked it off the main decision column.
    ("decision", 2130, 110, 2670, 230, "Was it cut off at", "the token limit?"),
    ("helper", 2130, 270, 2670, 475, "Helper agent:\nsummarize",
     "the cut-off\nattempt"),
    ("decision", 2130, 515, 2670, 720, "Did this round write\nfresh code?",
     "(and does the problem\nship stdin tests?)"),
    ("check", 2130, 760, 2670, 965, "Run solution.py\non the samples",
     "verdict goes to\nthe manager as\nground truth"),
]

# Elbow polylines; the arrowhead lands on the last point. Grouped by source box, in the
# same order as BOXES_CUR, so a wire's start is always the box just above it in this list.
# Several branch points are deliberately distinct even where two wires reach toward the
# same neighbourhood (finished's "no" entry into rounds vs. gate's "yes"; the three
# entries into writeans) so two wires never touch the exact same pixel and read as one.
# Each entry is (points, branch): branch is "yes"/"no" for a decision's two exits, coded
# by colour (DECISION_COLORS below) rather than a "yes"/"no" tag at every one of the 14
# branch wires -- 7 decisions x 2 branches was the most repeated, least informative text
# on the figure. The two branches that carry information a colour cannot ("out of
# rounds", "nothing changed") keep their text in LABELS_CUR; a bare "yes"/"no" doesn't
# reappear there. branch is None for a wire with only one destination, which needs no key.
WIRES_CUR = [
    ([(330, 175), (330, 215)], None),                                     # start -> plan
    ([(330, 380), (330, 420)], None),                                     # plan -> brainstorm
    ([(330, 625), (330, 665)], None),                                     # brainstorm -> manage
    ([(600, 770), (680, 770), (680, 232), (750, 232)], None),             # manage -> sample gate
    ([(1020, 355), (1020, 395)], "no"),                                   # gate: samples fine -> finished?
    ([(1290, 232), (1400, 232), (1400, 180), (1440, 180)], "yes"),        # gate: failed -> loop (rounds, upper)
    ([(1020, 500), (1020, 540)], "yes"),                                  # finished? yes -> saved?
    ([(1290, 447), (1360, 447), (1360, 310), (1440, 310)], "no"),         # finished? no -> rounds (lower)
    ([(1020, 745), (1020, 785)], "yes"),                                  # saved? yes -> skip
    ([(1290, 642), (1310, 642), (1310, 910), (1440, 910)], "no"),         # saved? no -> write-up
    ([(1020, 905), (1020, 1042), (1440, 1042)], None),                    # skip -> hand back
    ([(1710, 355), (1710, 395)], "yes"),                                  # rounds left? yes -> same task?
    ([(1440, 335), (1400, 335), (1400, 890), (1440, 890)], "no"),         # rounds left? no -> write-up
    ([(1440, 455), (1420, 455), (1420, 830), (1440, 830)], "yes"),        # same task? yes -> write-up
    ([(1710, 515), (1710, 555)], "no"),                                   # same task? no -> do it
    ([(1980, 657), (2060, 657), (2060, 170), (2130, 170)], None),         # worker -> cut off?
    ([(2400, 230), (2400, 270)], "yes"),                                  # cut off? yes -> summarise it
    ([(2130, 200), (2075, 200), (2075, 617), (2130, 617)], "no"),         # cut off? no -> fresh code (left entry)
    ([(2400, 475), (2400, 495), (2550, 495), (2550, 515)], None),         # digest rejoins the round (right entry)
    ([(2400, 720), (2400, 760)], "yes"),                                  # fresh code -> run samples
    ([(2670, 617), (2740, 617), (2740, 1145)], "no"),                     # no fresh code, no verdict -- joins the
                                                                           # return channel below, partway down it
    ([(2670, 862), (2740, 862), (2740, 1145), (330, 1145), (330, 875)], None),  # round -> back to manager, along
                                                                                 # the bottom, above the legend

]

# No labels at all: every decision branch is colour-only (the box's own heading states
# the question, the wire's colour states the answer), and the one non-branch wire that
# used to carry a "back to the manager" tag is now identifiable by simply following it.
LABELS_CUR = []

# Packed left-to-right with a uniform ~60px gap between entries: each x is the previous
# entry's end (swatch + 14px pad + its label's measured pixel width at FS_LEGEND) plus 60.
LEGEND_CUR = [("lead", "lead agent", 60), ("helper", "helper agent", 315),
              ("decision", "decision", 600), ("terminal", "start / end", 820),
              ("check", "sample-test run (no model call)", 1070),
              ("yes", "decision: yes", 1620), ("no", "decision: no", 1910)]

# No note band under this one: everything the 46710a5 notes explain -- the round budget,
# what the sample gate does and when it produces no verdict -- the paper now says in the
# S3.1 prose and the caption that run beside the figure, and a second account of it here
# is one more thing to keep in sync. `notes` is absent from its entry below, not empty.


# --- figure 2: multiagent.py @ 46710a5, the ORIGINAL scaffold (§2.3) --------------------
# Not the figure the paper includes; kept because it is the only drawing of the scaffold
# behind §2.3, and §3's list of v2-vs-original differences is read off the pair.

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

# Plain points, no branch colour: this diagram is not placed in the paper (see the
# module docstring), so it keeps the older text-label convention rather than picking up
# WIRES_CUR's colour coding for no compliance benefit.
WIRES_46 = [
    ([(434.5, 155), (434.5, 196)], None),                                # start -> plan
    ([(434.5, 264), (434.5, 304)], None),                                # plan -> brainstorm
    ([(434.5, 393), (434.5, 427)], None),                                # brainstorm -> manage
    ([(434.5, 522), (434.5, 565)], None),                                # manage -> finished?
    ([(331, 593), (210, 593), (210, 661)], None),                        # finished? yes
    ([(538, 593), (751, 593), (751, 654)], None),                        # finished? no
    ([(150, 736), (150, 803)], None),                                    # saved? yes -> skip
    ([(290, 736), (290, 925)], None),                                    # saved? no  -> write-up
    ([(751, 743), (751, 793)], None),                                    # rounds left? yes
    ([(751, 868), (751, 925)], None),                                    # same task? no -> do it
    ([(615, 698), (368, 698), (368, 925)], None),                        # rounds left? no
    ([(621, 830), (447, 830), (447, 925)], None),                        # same task? yes
    ([(888, 962), (924, 962), (924, 474), (571, 474)], None),            # worker -> back to manager
    ([(150, 859), (150, 1095), (245, 1095)], None),                      # skip -> hand back
    ([(381.5, 1000), (381.5, 1064)], None),                              # write-up -> hand back
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

# The source set the code tokens (solution.py, answer.md, ANSWER:, MULTIAGENT_MAX_ITERS)
# in a monospace face. Reproducing that with mathtext \mathtt puts math spacing around
# the punctuation -- "solution. py", "ANSWER : " -- so these are plain text instead. The
# wording is unchanged; only the face differs.
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
NOTE_DY = 21.5                          # only this figure carries a note band


# (title, canvas, boxes, wires, labels, legend + its y, and optionally notes + their first y)
# Width is the return lane (x=2740) plus a margin matching the left one; height is
# column 3's row (the deepest, now that every column packs from the same top) plus
# the legend and a slim margin under it.
FIG_CURRENT = dict(title=TITLE_CUR, size=(2820, 1250), y_pad=0, boxes=BOXES_CUR,
                   wires=WIRES_CUR, labels=LABELS_CUR, legend=LEGEND_CUR, legend_y=1180)

FIG_46710A5 = dict(title=TITLE_46, size=(1015, 1354), boxes=BOXES_46, wires=WIRES_46,
                   labels=LABELS_46, legend=LEGEND_46, legend_y=1156,
                   notes=NOTES_46, note_y0=1218)

LEGEND_PAD = 14         # swatch-to-label gap, px
LEGEND_GAP = 52         # gap between one entry's label and the next entry's swatch, px
LEGEND_ROW = 66         # row pitch when the row wraps, px; clears the taller swatch


def layout_legend(entries, width, y0):
    """Place the legend entries left to right, wrapping to a second row.

    The x values in LEGEND_* were spaced by hand for 5.5pt labels and collide at 7pt --
    the seven entries measure wider than the canvas on one row. Measuring each label
    off the real glyphs and wrapping on overflow means the row cannot overlap itself at
    whatever band FS_LEGEND is set to. Returns (kind, text, x, y) per entry.
    """
    fp = FontProperties(family=["Liberation Sans"])
    x0 = entries[0][2]
    avail = width - 2 * x0
    widths = [SWATCH_W + LEGEND_PAD
              + TextPath((0, 0), text, size=FS_LEGEND,
                         prop=fp).get_extents().width * DPI / 72
              for _, text, _ in entries]

    def rows_of(n):
        """Split into n contiguous rows of near-equal COUNT, or None if one won't fit."""
        per = math.ceil(len(entries) / n)
        chunks = [list(range(i, min(i + per, len(entries))))
                  for i in range(0, len(entries), per)]
        for c in chunks:
            if sum(widths[i] for i in c) + LEGEND_GAP * (len(c) - 1) > avail:
                return None
        return chunks

    # Balance by count rather than filling each row to the edge: greedy packing put six
    # entries on the first row and one orphan on the second, which reads as a mistake.
    chunks = next((r for n in range(1, len(entries) + 1) if (r := rows_of(n))), None)
    if chunks is None:                      # nothing fits; fall back to one per row
        chunks = [[i] for i in range(len(entries))]

    out = []
    for row, chunk in enumerate(chunks):
        x = x0
        for i in chunk:
            out.append((entries[i][0], entries[i][1], x, y0 + row * LEGEND_ROW))
            x += widths[i] + LEGEND_GAP
    return out


def stretch_y(figure, k=Y_STRETCH):
    """Scale every y in an authored figure by k; see the Y_STRETCH comment.

    The coordinates above stay as they were measured. This is the one place the
    diagram is made taller, so a box, the wire that leaves it and the label on that
    wire cannot drift apart -- they all move by the same factor, and nothing
    horizontal moves at all.
    """
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
    """Title -> filename stem, so the two never drift apart."""
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
    # y_pad extends the canvas ABOVE y=0 (title and content both start at or below 0) --
    # CUR's title and its "back to manager" wire/label both need headroom above the
    # topmost boxes, which start right at 0 for the reasons the BOXES_CUR comment gives.
    y_pad = figure.get("y_pad", 0)
    fig = plt.figure(figsize=(W / DPI, (H + y_pad) / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, -y_pad))  # y inverted: pixel coordinates, origin top-left
    ax.axis("off")
    ax.set_facecolor(t["surface"])

    # The title's y is the one coordinate not in the figure dict, so stretch_y cannot
    # reach it; scaled here so it keeps its gap to the boxes it sits above.
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
            # Heading sits in the upper part of the box, subtext below it, the two
            # blocks stacked and centred as a unit rather than pinned at fixed fractions
            # of the box: a fraction tuned for a 1-2 line head/sub does not generalise to
            # the 1-3 line combinations this figure's rewrapped text now uses. Each
            # block's height is measured in points from its own line count and linespacing,
            # then converted to pixels at this figure's DPI -- the same unit the box
            # coordinates are already in.
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
        lw = 2.4 if branch else 1.2    # decision branches read as the thicker, coloured
                                        # lines; plain sequential wires stay hairline black
        if len(pts) > 2:               # elbows first, so the head sits on the last leg only
            ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                    color=color, lw=lw, ls=style, solid_capstyle="butt",
                    solid_joinstyle="miter", dash_capstyle="butt", zorder=1)
        # The arrow head's own leg carries the dash too, so a branch is one style end to
        # end rather than turning solid at the tip.
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=style, shrinkA=0, shrinkB=0,
                                    mutation_scale=9), zorder=1)

    for text, x, y in figure["labels"]:
        ax.text(x, y, text, ha="center", va="center", fontsize=FS_LABEL, color=t["ink"])

    for kind, text, x, row_y in layout_legend(figure["legend"], W, figure["legend_y"]):
        if kind in DECISION_COLORS:    # a line swatch for a branch colour, not a box fill
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
        for theme in ("light",):  # no dark twin -- the paper only \inputs light
            draw(figure, theme,
                 save=os.path.join(PLOTS, f"{slug(figure['title'])}_{theme}.pdf"))


if __name__ == "__main__":
    main()
