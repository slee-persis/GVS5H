#!/usr/bin/env python3
"""Write paper/fig-*.tex: the three two-panel figures, captions and all.

The charts carry only what has to sit beside a mark. Everything else -- how to read
the fills, what the test was, the pairwise Tukey p values, the emit rates -- is
caption text, which LaTeX sets at real \\footnotesize instead of the ~5pt an in-chart
note block could manage at this canvas size.

Each condition module owns its own CAPTION and notes(stats); this script only turns
them into LaTeX and pairs them with the PDF the plot scripts wrote. Numbers
inside a note are read off `stats`, so a re-run updates the paper -- nothing here is
transcribed by hand.

    uv run --with matplotlib --with numpy --with scipy python \\
        paper_plot_script/make_figures_tex.py

Run it after the plot scripts; paper.tex \\inputs the files it writes.
"""
import os
import re

import plot_16k_reason_off_5_pass as p16
import plot_128k_reason_off_1_pass as p128off
import plot_128k_reason_on_1_pass as p128on
import plot_4new_5pass_reason_on as p4new
import plot_cost_5_pass as pcost
import plot_cost_vs_score as pcvs
from plot_16k_reason_off_5_pass import slug, tukey, tukey_sentence
from plot_bars import by_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)    # repo root, one level up from paper_plot_script/
PAPER = os.path.join(ROOT, "paper")

# The paper's figures, in the order they appear there, then the ones written but not
# yet \input anywhere. `tukey` is off where there is no across-model HSD to report: the
# Qwen3.8-Max vs Fable-5 figure has two models and one of them has a single arm, so its
# three comparisons are the paired tests in its own notes.
FIGURES = [
    # The scale panel of each -- the same numbers against parameter count, which is the
    # only view that shows how the two curves move with model size. Extended Data, with
    # the note block suppressed: those notes describe the tests, which the bar panel in
    # the main text carries.
    dict(mod=p128on, out="fig-128k-reason-on-scale.tex", panels=("dots",), tukey=False,
         notes=False, label="fig:scale-on", caption="\\textbf{Accuracy by model scale, reasoning on, 128k, "
         "one pass.} The same numbers as Figure~\\ref{fig:scale-bars}a, against parameter count. "
         "Dashed = single call, solid = with manager; the block under each tick gives that model's "
         "$\\Delta$ and Tukey group, and the legend each arm's gain from the smallest model to the largest."),
    dict(mod=p16, out="fig-16k-reason-off-scale.tex", panels=("dots",), tukey=False,
         notes=False, label="fig:scale-16k", caption="\\textbf{Accuracy by model scale, reasoning off, 16k, "
         "five passes.} The same numbers as Figure~\\ref{fig:scale-bars}b, against parameter count; the line "
         "through each mark is the 95\\% CI across the five passes. "
         "Dashed = single call, solid = with manager, and the bracket at a pair grades its Holm-corrected p; "
         "the block under each tick gives that model's $\\Delta$, p and Tukey group, and the legend each "
         "arm's gain from 9B to 2.8T."),
    dict(mod=p128off, out="fig-128k-reason-off-scale.tex", panels=("dots",), tukey=False,
         notes=False, label="fig:scale-off", caption="\\textbf{Accuracy by model scale, reasoning off, 128k, "
         "one-pass control.} The same numbers as Figure~\\ref{fig:scale-bars}c, against parameter count. "
         "Dashed = single call, solid = with manager; the block under each tick gives that model's "
         "$\\Delta$ and Tukey group, and the legend each arm's gain from the smallest model to the largest."),
    # tukey=False: the four pinned-backend models are not a scale series, and three of
    # them have undisclosed sizes, so a cross-model HSD would be uninterpretable.
    # Bars only: with four models and no scale axis the dot panel showed the same eight
    # numbers in the same left-to-right order, so it was a second rendering rather than a
    # second view. The CIs it carried are on the bars too.
    dict(mod=p4new, out="fig-4new-5pass.tex", tukey=False, panels=("bars",), half=True,
         label="fig:main",
         captions=("Accuracy per condition, with 95\\% CIs across the 5 passes.",
                   "Accuracy per condition, with 95\\% CIs across the 5 passes.")),
    # plot_q38_vs_fable5_5_pass is deliberately absent: its four manager-vs-Fable-5 bars
    # and all three of its p values are already in fig:main's long brackets and caption,
    # so the paper dropped it rather than print the comparison twice. The script is kept
    # and run by hand -- it is still the place those numbers are computed -- but nothing
    # here imports it, so a default run writes no chart for it.
    # Single panel: this chart is one scatter, not a dots/bars pair.
    dict(mod=pcvs, out="fig-cost-vs-score.tex", tukey=False, panels=("dots",),
         label="fig:cost-vs-score",
         captions=("Cost against accuracy, all seven pinned-backend arms.", "")),
    # one panel, then the module's own table, then a one-line caption. The tables are
    # written to their own file so paper.tex can \input the cost-vs-score figure between
    # the chart and them: a barrier flushes floats in declaration order, and with the two
    # tables ahead of it Figure 3 was pushed onto a sheet of its own a page later.
    dict(mod=pcost, out="fig-cost.tex", tukey=False, table=True, half=True,
         label="fig:cost", tables_label="tab:cost",
         tables_out="fig-cost-tables.tex"),
]

# The three model-scale bar charts (formerly Figures 3-5) as one figure instead of
# three: an 8-page Article allows 5-6 main display items, and three near-identical
# single-panel bar charts is what panels are for. Each keeps its plot_bars.py drawing
# and its own condition module for stats/notes; only the presentation combines. The
# compact PNGs plot_bars.draw_compact() writes are a second design, not a resize of the
# ones above -- see that function's docstring for why a resize could not work here.
COMBINED_SCALE_BARS = dict(
    out="fig-scale-bars.tex",
    label="fig:scale-bars",
    conds=[
        (p128on, "128k max tokens, reasoning on, one pass per condition."),
        (p16, "16k max tokens, reasoning off, five passes per condition."),
        (p128off, "128k max tokens, reasoning off, one pass per condition."),
    ],
)

# The notes are written as plain prose with real Unicode; these are the characters
# that would otherwise be a LaTeX special or a missing glyph. Order matters: the
# backslash has to go first, or it would escape the escapes.
TEX = [
    ("\\", "\\textbackslash{}"),
    ("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("_", "\\_"),
    ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ("Δ", "$\\Delta$"), ("α", "$\\alpha$"), ("×", "$\\times$"), ("−", "$-$"),
    ("→", "$\\to$"), ("≥", "$\\ge$"), ("≤", "$\\le$"), ("<", "$<$"), (">", "$>$"),
    ("—", "-"), ("–", "--"), ("’", "'"),
    # Currency. A note cannot write a bare "$" -- tex() treats dollar signs as math
    # delimiters (so fmt_p_num's mathtext survives), and a note about prices would pair
    # them up and set half its words in math italics. Notes write "¤12.34"; this maps it
    # to a real escaped dollar. It has to sit AFTER the backslash rule above, or the
    # backslash it introduces would itself be escaped.
    ("¤", "\\$"),
]


_SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def tex(text):
    """Prose -> LaTeX. Straight double quotes become a proper open/close pair.

    Spans between dollar signs are passed through untouched: fmt_p_num() writes a
    small p as matplotlib mathtext ($2\\times10^{-7}$), which is already LaTeX, and
    escaping it would print the backslashes. fmt_p_num's plain-text form instead
    writes the exponent with Unicode superscripts (its docstring says why); pdflatex
    has no glyphs for those, so they are folded back into math here first.
    """
    text = re.sub("[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+",
                  lambda m: "$^{%s}$" % m.group().translate(_SUP), text)
    out, opening = [], True
    for i, span in enumerate(text.split("$")):
        if i % 2:                                  # inside math: already LaTeX
            out.append(f"${span}$")
            continue
        for a, b in TEX:
            span = span.replace(a, b)
        for ch in span:
            if ch == '"':
                out.append("``" if opening else "''")
                opening = not opening
            else:
                out.append(ch)
    return "".join(out)


def model_summary(stats, letters):
    """Delta and Tukey group per model, compact -- the numbers draw_compact() dropped
    from its own per-model block to fit three panels under Nature's 170mm depth cap.
    Read off `stats`/`letters`, the same values the chart would have printed, so this
    cannot drift from the run the way a hand-typed table could."""
    parts = [f"{st['label']} \u0394{st['delta']:+.1f} (group {letters[i]})"
             for i, st in enumerate(stats)]
    return ", ".join(parts) + "."


def combined_scale_bars_tex(entry):
    """The three model-scale bar charts as one (a)/(b)/(c) figure.

    Each panel is plot_bars.draw_compact()'s PNG for that condition; the caption states
    the shared fill/CI conventions once, then each panel's own notes()  (already
    concise -- see plot_16k_reason_off_5_pass.py's CAPTION comment) plus the per-model
    Delta/group summary the compact panel had no room to print itself.
    """
    letters_tag = "abc"
    panel_blocks, note_blocks = [], []
    for i, (mod, condition_line) in enumerate(entry["conds"]):
        stats = mod.compute()
        _, letters = tukey(stats)
        # the panels draw in this order too (plot_bars.draw_compact), so the caption's
        # per-model summaries run down the panel rather than against it
        stats, letters = by_score(stats, letters)
        stem = slug(mod.TITLE)
        sep = "\\hfill" if i < len(entry["conds"]) - 1 else ""
        panel_blocks.append(
            f"\\begin{{subfigure}}[t]{{0.32\\linewidth}}\n"
            f"  \\centering\n"
            f"  \\panelplot{{\\plotdir/{stem}_bars_compact_light.pdf}}\n"
            f"  \\caption{{{tex(condition_line)}}}\n"
            f"\\end{{subfigure}}{sep}")
        # The shared caption lead already states the fill/Delta convention once for all
        # three panels; strip it from a panel's own notes so the legend stays short
        # instead of saying it twice.
        shared = ("Fill: light = single call (one call, no tools), dark = with manager; "
                  "Δ = manager − single, in percentage points.")
        notes = [n[len(shared):].lstrip() if n.startswith(shared) else n
                 for n in mod.notes(stats)]
        body = " ".join(tex(n) for n in notes if n)
        body += " " + tex(model_summary(stats, letters))
        note_blocks.append(f"({letters_tag[i]}) {body}")
    panels = "\n".join(panel_blocks)
    caption = (
        "\\textbf{Manager vs.\\ single call, three conditions.} "
        "Models run top to bottom by single-call score, best first, so a model's row "
        "differs between panels; the top bar of each pair is the manager. "
        "Fill: light = single call (one call, no tools), dark = with manager; "
        "$\\Delta$ = manager $-$ single, in percentage points. The line through a bar "
        "is the 95\\% CI across its passes, where the condition ran more than one. "
        "A panel prints its two bar values and nothing else; the $\\Delta$ and Tukey HSD "
        "group letter quoted below per model are what it leaves out (shared letter = not "
        "significantly different across the models of that condition, $\\alpha = 0.05$). "
        + " ".join(note_blocks) + "\n")
    return (
        "% Generated by paper_plot_script/make_figures_tex.py -- do not edit.\n"
        "% The caption text lives in each condition's plot script (CAPTION/notes) and\n"
        "% here (the shared lead and the per-model summaries); re-run both to update.\n"
        "\\begin{figure}[p]\n"
        "\\centering\n"
        f"{panels}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{entry['label']}}}\n"
        "\\end{figure}\n")


def figure_tex(fig, stats, pmat):
    """One float page: dot panel, bar panel, then the caption with the notes.

    `panels` in the registry selects which of the two are drawn. A figure asking for one
    of them gets no subfigure wrapper and no (a)/(b) sublabel -- a lone panel labelled (b)
    reads as a figure with a missing half.
    """
    mod = fig["mod"]
    stem = slug(mod.TITLE)
    notes = [tex(n) for n in mod.notes(stats)]
    if pmat is not None:
        notes.append(tex(tukey_sentence(stats, pmat)))
    caps = fig.get("captions", ("Accuracy by model scale.",
                                "The same numbers as paired bars."))
    wanted = fig.get("panels", ("dots", "bars"))
    ext = ".pdf"
    available = {"dots": (f"{stem}_light{ext}", caps[0]),
                 "bars": (f"{stem}_bars_light{ext}", caps[1])}
    chosen = [available[k] for k in wanted]
    if len(chosen) == 1:
        panels = f"\\panelplot{{\\plotdir/{chosen[0][0]}}}"
    else:
        panels = "\n".join(
            f"\\begin{{subfigure}}{{\\linewidth}}\n"
            f"  \\centering\n"
            f"  \\panelplot{{\\plotdir/{png}}}\n"
            f"  \\caption{{{cap}}}\n"
            f"\\end{{subfigure}}{sep}"
            for (png, cap), sep in zip(chosen, ("\\\\[0.7ex]", "")))
    # Two panels plus these notes fill a page, so those go on a [p] float page. One panel
    # does not: on [p] it prints a half-empty sheet, and it is small enough to sit in the
    # text near the paragraph that cites it. `h` leads so it does that -- follows the
    # prose -- rather than being thrown to the top or foot of the page and leaving a gap
    # where it was cited.
    # `where` in the registry overrides this: two single-panel figures cited by the
    # same paragraph only share a page as a float PAGE, since h/t placement takes
    # them one at a time and there is no body text after them to fill the rest.
    where = fig.get("where", "[p]" if len(chosen) > 1 else "[htbp]")
    # half=True: the chart is designed at single-column width -- a colfigure float
    # (one column of the published layout) holding a \halfplot
    # chartfigure carries its own layout-dependent placement (see the preamble), so
    # `where` is not written for a half-width chart
    env = "chartfigure" if fig.get("half") else "figure"
    if fig.get("half"):
        panels = panels.replace("\\panelplot", "\\halfplot")
        where = ""
    return (
        "% Generated by paper_plot_script/make_figures_tex.py -- do not edit.\n"
        "% The caption text lives in that chart's plot script, next to the numbers it\n"
        "% describes; re-run the script to update this file.\n"
        f"\\begin{{{env}}}{where}\n"
        "\\centering\n"
        f"{panels}\n"
        f"\\caption{{{fig.get('caption', mod.CAPTION)} "
        # The notes run on from the caption itself, as one paragraph: they are prose
        # about the same chart, and a note per line reads as a list of unrelated
        # remarks. They used to carry \footnotesize to set them a size below the
        # title sentence; iclr2027_conference.sty defines \footnotesize and \small
        # as the SAME 9pt, and \captionsetup already puts the caption at \small, so
        # that step never rendered. Dropped rather than deepened: the template asks
        # for no font-size changes, and the lead sentence is bold, which separates it.
        + " ".join(notes if fig.get("notes", True) else []) + "}\n"
        f"\\label{{{fig['label']}}}\n"
        f"\\end{{{env}}}\n")


def sources(mod):
    """The rate-source line, set under the rate table, or "" for a module without rates.

    It goes with the table rather than under the chart because the rates are a column of
    that table: the reference belongs beside the numbers it publishes, not beside bars
    that are those numbers already multiplied out. The chart no longer carries it at all -
    both priced figures used to repeat the same line under their panels.
    """
    if not getattr(mod, "SOURCES", None):
        return ""
    # \par first: a tabular is an hbox, so without it the line runs on beside the table
    # instead of under it.
    return f"\\par\\vspace{{0.6ex}}\n{{\\footnotesize {mod.SOURCES}}}\n"


def tabular(header, spec, rows):
    """One booktabs tabular. The rows arrive as LaTeX from the module, so nothing here
    is escaped or reformatted."""
    head = " & ".join(f"\\textbf{{{c}}}" for c in header)
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return (f"\\begin{{tabular}}{{{spec}}}\n"
            "\\toprule\n"
            f"{head} \\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n")


def table_figure_tex(fig, stats):
    """The panel as its own figure float, and the module's tables as their own.

    Returns the two as separate strings, one file each, so the paper can put another
    float between them.

    Measured against a 650pt text block: panel 243pt, caption 106pt, tables 255pt. Kept in
    one float that is 605pt -- 93% of a page -- which only [p] can ever place, so it always
    took a sheet of its own and could never sit beside the paragraphs that read it. Split,
    the figure is 54% and fits at the top of a text page, and the tables (39%) float
    separately to wherever they land.

    Booktabs and the column specs follow the tables already in paper.tex.
    """
    mod = fig["mod"]
    head = (
        "% Generated by paper_plot_script/make_figures_tex.py -- do not edit.\n"
        "% The caption text and the table numbers live in that chart's plot script;\n"
        "% re-run the script to update this file.\n")
    # chartfigure = colfigure with a layout-dependent placement, defined in the
    # preamble (see the comment there for why the branch cannot live here)
    env = "chartfigure" if fig.get("half") else "figure"
    # Placement differs by layout, so the two are branched in the .tex itself: TeX
    # skips the untaken branch while expanding, so exactly one \begin survives and
    # pairs with the single \end below.
    #
    # One column (referee copy): [!hb]. A TOP float on the page it is declared on
    # lands above the "What the scaffold costs" heading and above the tail of the
    # section before it -- the figure printing before the section it belongs to. So
    # `t` is excluded outright; `h` leads, so the figure follows the paragraph that
    # cites it instead of dropping to the foot of the page and leaving a gap behind.
    #
    # Two columns (published): [!tb]. `t` is not only safe here but necessary. It is
    # safe because a float declared part-way down the left column cannot be placed
    # above its own declaration point -- the earliest top slot is the RIGHT column,
    # already past the heading. It is necessary because at 68mm plus this caption the
    # float no longer fits in whatever the declaring column has left, and with `h`/`b`
    # alone it defers to the next page, printing after its section has ended.
    #
    # The ! waives \bottomfraction/\topfraction, which would otherwise reject a float
    # this tall.
    figure = (
        f"\\begin{{{env}}}\n"
        "\\centering\n"
        f"\\{'halfplot' if fig.get('half') else 'panelplot'}"
        f"{{\\plotdir/{slug(mod.TITLE)}_light.pdf}}\n"
        f"\\caption{{{fig.get('caption', mod.CAPTION)} "
        # This module's notes are authored as LaTeX already -- money() emits an
        # escaped dollar, fmt_p_tex() emits math -- so they bypass tex(), whose
        # span-splitting on "$" would mangle the escaped ones.
        + " ".join(mod.notes(stats)) + "}\n"
        f"\\label{{{fig['label']}}}\n"
        f"\\end{{{env}}}\n")
    # Real table floats rather than \captionof inside the figure: they carried table
    # numbers before and still do, but now they can be placed independently of the chart.
    # h before b, and no t: paper.tex declares these inside \S2.2, and a top float lands
    # above that heading and above the tail of \S2.1, so the cost tables would print
    # before the section that introduces them. `h` puts them straight under the opening
    # paragraph that quotes their numbers; `b` is the fallback, and pinning them there
    # outright left a half-page of white between the paragraph and Table 3. The ! waives
    # \bottomfraction (30%), which the pair at 39% of the block would otherwise fail.
    # The rate line goes under the first table, which is the one with the rate column;
    # \vspace* on the later ones keeps that footnotesize line from crowding the next
    # caption when the two tables land on the same page (\floatsep does not apply to
    # floats set `h`, so the space has to travel inside the box).
    tables = "\n".join(
        "\\begin{table}[!hb]\n"
        + ("" if i == 0 else "\\vspace*{2ex}\n")
        + "\\centering\n"
        "\\small\n"
        f"\\caption{{{cap}}}\n"
        f"\\label{{{fig['tables_label']}{'' if i == 0 else i}}}\n"
        + tabular(header, spec, rows(stats))
        + (sources(mod) if i == 0 else "") +
        "\\end{table}\n"
        for i, (header, spec, rows, cap) in enumerate(mod.TABLES))
    return head + figure, head + tables


def main():
    path = os.path.join(PAPER, COMBINED_SCALE_BARS["out"])
    with open(path, "w") as fh:
        fh.write(combined_scale_bars_tex(COMBINED_SCALE_BARS))
    print("wrote", path)

    for fig in FIGURES:
        stats = fig["mod"].compute()
        if fig.get("table"):
            figure, tables = table_figure_tex(fig, stats)
            for name, body in ((fig["out"], figure), (fig["tables_out"], tables)):
                path = os.path.join(PAPER, name)
                with open(path, "w") as fh:
                    fh.write(body)
                print("wrote", path)
            continue
        pmat = tukey(stats)[0] if fig.get("tukey", True) else None
        path = os.path.join(PAPER, fig["out"])
        with open(path, "w") as fh:
            fh.write(figure_tex(fig, stats, pmat))
        print("wrote", path)


if __name__ == "__main__":
    main()
