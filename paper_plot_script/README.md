# paper_plot_script

The scripts that draw every PNG in [paper/plots/](../paper/plots/). Each reads the raw
result JSON in [runs/](../runs/), recomputes the numbers itself, and writes a
`_light` and a `_dark` twin of its chart; the paper `\input`s the `_light` ones.

Run from the repo root:

```
uv run --with matplotlib --with numpy --with scipy python paper_plot_script/<script>.py
```

| script | figure |
| --- | --- |
| `plot_16k_reason_off_5_pass.py` | manager vs single call across model scale, 16k, reasoning off, 5 passes |
| `plot_128k_reason_off_1_pass.py` | same at 128k, reasoning off, 1 pass |
| `plot_128k_reason_on_1_pass.py` | same at 128k, reasoning on, 1 pass |
| `plot_bars.py` | the `_bars` panel of the three above |
| `plot_4new_5pass_reason_on.py` | manager vs single call, four pinned-backend models |
| `plot_cost_5_pass.py` | what one pass costs, single call vs manager against Fable-5 |
| `plot_cost_vs_score.py` | what accuracy costs, all seven pinned-backend arms |
| `plot_agent_loop_flowchart.py` | the two manager-loop diagrams |
| `plot_q38_vs_fable5_5_pass.py` | the manager arms against Fable-5 (not a paper figure; see below) |

Then `make_figures_tex.py` writes [paper/fig-*.tex](../paper/) — the two-panel
figure environments, captions and all. Run it after the charts: each condition
module owns its own `CAPTION` and `notes(stats)`, and the numbers inside a note
are read off `stats`, so a re-run updates the paper rather than anyone
transcribing a p value by hand. The paper `\input`s what it writes.

`plot_q38_vs_fable5_5_pass.py` draws no figure the paper uses — its four
manager-vs-Fable-5 bars and all three p values are already in Figure 1's
brackets and caption. It is kept because it is still where those numbers are
computed, and `make_figures_tex.py` imports it.

`plot_16k_reason_off_5_pass.py` is the shared module: themes, sizing, `slug()`,
`wrap_title()`, the CI and permutation helpers. The others import from it, so it
has to stay a sibling. `plot_cost_vs_score.py` also imports `SOURCES`/`TOKENS`
from `plot_cost_5_pass.py`.

`extract_tokens.py` is not a chart. It walks the run JSON for per-problem,
per-pass token counts and writes [runs/per_problem_tokens.json](../runs/per_problem_tokens.json),
which both cost charts read. That file is committed; rerun this only if the runs
change.
