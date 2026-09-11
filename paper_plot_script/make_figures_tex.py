#!/usr/bin/env python3
"""Write paper/fig-*.tex: the three two-panel figures, captions and all."""
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
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper")

FIGURES = [
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
    dict(mod=p4new, out="fig-4new-5pass.tex", tukey=False, panels=("bars",), half=True,
         label="fig:main",
         captions=("Accuracy per condition, with 95\\% CIs across the 5 passes.",
                   "Accuracy per condition, with 95\\% CIs across the 5 passes.")),
    dict(mod=pcvs, out="fig-cost-vs-score.tex", tukey=False, panels=("dots",),
         label="fig:cost-vs-score",
         captions=("Cost against accuracy, all seven pinned-backend arms.", "")),
    dict(mod=pcost, out="fig-cost.tex", tukey=False, table=True, half=True,
         label="fig:cost", tables_label="tab:cost",
         tables_out="fig-cost-tables.tex"),
]

COMBINED_SCALE_BARS = dict(
    out="fig-scale-bars.tex",
    label="fig:scale-bars",
    conds=[
        (p128on, "128k max tokens, reasoning on, one pass per condition."),
        (p16, "16k max tokens, reasoning off, five passes per condition."),
        (p128off, "128k max tokens, reasoning off, one pass per condition."),
    ],
)

TEX = [
    ("\\", "\\textbackslash{}"),
    ("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("_", "\\_"),
    ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ("Δ", "$\\Delta$"), ("α", "$\\alpha$"), ("×", "$\\times$"), ("−", "$-$"),
    ("→", "$\\to$"), ("≥", "$\\ge$"), ("≤", "$\\le$"), ("<", "$<$"), (">", "$>$"),
    ("—", "-"), ("–", "--"), ("’", "'"),
    ("¤", "\\$"),
]


_SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def tex(text):
    text = re.sub("[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+",
                  lambda m: "$^{%s}$" % m.group().translate(_SUP), text)
    out, opening = [], True
    for i, span in enumerate(text.split("$")):
        if i % 2:
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
    parts = [f"{st['label']} \u0394{st['delta']:+.1f} (group {letters[i]})"
             for i, st in enumerate(stats)]
    return ", ".join(parts) + "."


def combined_scale_bars_tex(entry):
    letters_tag = "abc"
    panel_blocks, note_blocks = [], []
    for i, (mod, condition_line) in enumerate(entry["conds"]):
        stats = mod.compute()
        _, letters = tukey(stats)
        stats, letters = by_score(stats, letters)
        stem = slug(mod.TITLE)
        sep = "\\hfill" if i < len(entry["conds"]) - 1 else ""
        panel_blocks.append(
            f"\\begin{{subfigure}}[t]{{0.32\\linewidth}}\n"
            f"  \\centering\n"
            f"  \\panelplot{{\\plotdir/{stem}_bars_compact_light.pdf}}\n"
            f"  \\caption{{{tex(condition_line)}}}\n"
            f"\\end{{subfigure}}{sep}")
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
    where = fig.get("where", "[p]" if len(chosen) > 1 else "[htbp]")
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
        + " ".join(notes if fig.get("notes", True) else []) + "}\n"
        f"\\label{{{fig['label']}}}\n"
        f"\\end{{{env}}}\n")


def sources(mod):
    if not getattr(mod, "SOURCES", None):
        return ""
    return f"\\par\\vspace{{0.6ex}}\n{{\\footnotesize {mod.SOURCES}}}\n"


def tabular(header, spec, rows):
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
    mod = fig["mod"]
    head = (
        "% Generated by paper_plot_script/make_figures_tex.py -- do not edit.\n"
        "% The caption text and the table numbers live in that chart's plot script;\n"
        "% re-run the script to update this file.\n")
    env = "chartfigure" if fig.get("half") else "figure"
    figure = (
        f"\\begin{{{env}}}\n"
        "\\centering\n"
        f"\\{'halfplot' if fig.get('half') else 'panelplot'}"
        f"{{\\plotdir/{slug(mod.TITLE)}_light.pdf}}\n"
        f"\\caption{{{fig.get('caption', mod.CAPTION)} "
        + " ".join(mod.notes(stats)) + "}\n"
        f"\\label{{{fig['label']}}}\n"
        f"\\end{{{env}}}\n")
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
