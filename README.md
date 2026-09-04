# GVS5H: Five Qwen3.8-27B Models Match Claude Fable 5 on LiveCodeBench Hard

**Fable 5 Level Coding for a Fifth the Price - or on a Single GPU**

Everything behind the paper's numbers: the paper itself, both versions of the
manager–worker scaffold, the benchmark harness they call, and the complete transcripts and
workspaces of every run the paper reports.

Every figure in the paper was recomputed from the copies in `runs/` before this bundle was
written, and reproduces exactly — see §4.

**No credentials are included.** `escalation/.env`, `.env.groq` and `.env.openrouter` are
excluded, as are any `*.key` / `.credentials*` files, and the tree was scanned for
key-shaped strings before packing. Model routing, caps and reasoning mode are all
environment-driven; supply your own keys.

---

## 1. `paper/`

`zero_shot_self_orchestration_with_ledger_based_control_for_improved_llm_coding_performance_2026-08-25.tex` and the PDF built from
it, plus the generated `fig-*.tex` figure files, `references.bib`, and, in `plots/`, every
PNG the document references.

The manuscript is formatted for **ICLR 2027**. `iclr2027_conference.sty`, `.bst`,
`fancyhdr.sty` and `natbib.sty` are checked in beside it so it builds from a clean clone;
the upstream `iclr-2027-style-files.zip` and the `iclr2027/` directory it unpacks to are
gitignored.

It builds in place:

```bash
latexmk -pdf zero_shot_self_orchestration_with_ledger_based_control_for_improved_llm_coding_performance_2026-08-25.tex
```

**One line differs from the repository copy**: line 24, `\newcommand{\plotdir}{plots}`,
points at `./plots` here instead of the repo's `../escalation/plots`.

**Anonymity is a toggle**: `\finalversionfalse` on line 21 builds the double-blind
submission — anonymous author block, review line numbers, no repository URLs.
`\finalversiontrue` restores the author block, the acknowledgements, the
author-contribution statement and the links. Submit with it false.

Main text (§1–§7) is 9 pages, ICLR's strict limit. The AI use, ethics and reproducibility
statements, the references and the appendices do not count against it.

The `fig-*.tex` files are generated, not hand-written — captions and footnotes live in the
plot script that draws each chart, and `run_bench_script/make_figures_tex.py` turns them
into LaTeX. Edit a caption in the plot script and re-run that, never in `fig-*.tex`.
`plot_bars.py` draws the three compact panels of Figure 5; the other scripts draw one
figure each — Figures 1–4 and 6–8.

Cross-references are `\label`/`\ref`, including from the generated captions, so the float
and section numbering survives reordering. The `label=` keys in `make_figures_tex.py` and
the `\cite` keys in `references.bib` are load-bearing: renaming one silently breaks a
generated caption.

---

## 2. `codebase/`

### the original scaffold
Produced the **OpenRouter-served model set** (§5.4).

### the updated scaffold
Produced the **seven first-party arms** (§5.1–§5.3).

The paper (Appendix A) names four differences, all acting on the manager arm alone. They are
verifiable directly in `escalation/multiagent.py`:

| | v1 | v2 |
|---|---|---|
| round budget (`MULTIAGENT_MAX_ITERS`) | 4 | 10 |
| sample-test verifier (§3 step 5) | absent | present |
| cut-off summarizer | absent | present |
| size bounds on workspace files | absent | present |

Because the single-call baseline is one call under either version, manager-minus-single
deltas are **not strictly comparable** between the two sets. That is why the paper reports
the OpenRouter models as their own condition (§5.4) rather than pooling them with §5.1.

Two further differences worth knowing, which the paper does not enumerate:

- **Problem selection.** v1 selects the latest 100 hard problems at run time
  (`--lcb 100`). Id pinning arrived later, so `escalation/lcb100_hardest_v6.json` — the
  frozen list of 100 `question_id`s — exists only under `v2-current/`. Both resolve to the
  same 100 problems; the pinned file makes it reproducible rather than date-dependent.
- **One post-paper fix is present in v2.** `orchestrator.py` now compares a suspected
  provider clamp against the cap actually sent rather than the configured cap. This was
  written *after* the runs and did not affect them (it only bites when a 400/context error
  shrinks the cap mid-run, which happened to a model not in the paper).

Two analysis tools under `v2-current/escalation/` were written for this revision of the
paper and are worth calling out:

- **`regrade.py`** re-scores stored generations against the *fixed* evaluator without
  re-running any model. Appendix C explains why this was necessary; §4 below shows how to repeat
  it. It writes `<name>.regraded.json` beside each input and never modifies the input.
- **`capmatch_q38.py`** replays Qwen3.8-27B's single-call generations against a 128k output
  cap, token-exactly, using the serving stack's own tokenizer. That arm was generated at
  250k while its manager twin ran at 128k; the paper reports it cap-matched (Appendix B) so the
  pair is like-for-like. It writes `*.cap128k.json`.

### `livecodebench/`
The benchmark harness both versions import for dataset loading, code extraction and
hidden-test grading (`escalation/run_bench.py` puts it on `sys.path`). Shared by v1 and
v2 — neither scaffold vendors its own copy.

**This copy carries the evaluator fix of Appendix C.** Upstream's stdin mock implemented
`MockBuffer.readline()` as a stateless expression that returned line 1 on every call, so
any solution reading multi-line input through `sys.stdin.buffer.readline()` was scored
wrong however correct it was. See `lcb_runner/evaluation/testing_util.py`. Every number in
the paper is reported after fixing it.

---

## 3. `runs/` — every run the paper reports

| directory | condition | paper location | workspaces |
|---|---|---|---|
| `firstparty-128k-reasoning-on-5pass/` | 128k cap, reasoning on, ×5, first-party APIs — Qwen3.8-27B, GPT-5.6-Luna, GPT-5.6-Terra | §5.1–§5.3, Figures 2–4 | 3,200 |
| `fable5-128k-reasoning-on-5pass/` | 128k cap, reasoning on, ×5, Anthropic API — Claude Fable 5, single-call only | §5.1–§5.3, Figures 2–4 | 500 |
| `16k-reasoning-off-5pass/` | 16k cap, reasoning off, ×5 | §5.4, Figures 5b and 7 | 4,500 |
| `128k-reasoning-off-1pass/` | 128k cap, reasoning off, ×1 | §5.4, Figures 5c and 8 | 800 |
| `128k-reasoning-on-1pass/` | 128k cap, reasoning on, ×1 | §5.4, Figures 5a and 6 | 901 |
| `q9-reasoning-on-archived/` | Qwen3.5-9B with thinking on | §6.1 (limitation) | — |

`per_problem_tokens.json` at the top of `runs/` holds the per-problem, per-pass token
counts every dollar figure in §5.2 is computed from, including calls that were retried and
discarded — those were generated and would be billed. Regenerate it with
`run_bench_script/extract_tokens.py`.

The six symlinks in `runs/` (`4models-1pass-reason-on`, `fable5-5pass-single`,
`models-lcb-5pass`, `128k-clean`, `results_think_high`, `ws_think_high`) are the names the
plot scripts address these directories by. They exist so every script runs unmodified;
extract with a tool that preserves symlinks, or re-point the paths at the top of each
script.

### What a run directory holds

`results/` (graded JSON, one file per config/pass) and `ws/` (per-problem workspaces). A
workspace holds what the roles actually read and wrote:

```
<hash>/task.md          the problem statement
<hash>/plan.md          the manager's overarching plan
<hash>/tasks.json       the task list [{id, desc, status, result}]
<hash>/notes.md         accumulated ideas / findings / partial proofs
<hash>/solution.py      the code that was graded
<hash>/transcript.jsonl every model call, in order, with its role
```

Workspace directory names are content hashes, not question ids. To map one to a problem,
match `solution.py` against the `code` field of the corresponding record in `results/`.

Result records carry `question_id`, `code`, `passed`, and — in the 128k conditions —
`status` ∈ {ok, truncated, empty_stop, empty, error}, `finish_reason` and
`completion_tokens`. The 16k ×5 runs predate that instrumentation and carry only the
extracted code and its pass/fail, which is why §5.4's 16k column is a broader *no-code*
count rather than a strict truncation count.

### Which file is the graded one

Three suffixes appear side by side, and the paper reports the last of them:

- `<name>.json` — as generated, scored by the evaluator as it stood at run time.
- `<name>.cap128k.json` — Qwen3.8-27B's single arm only: the same generations truncated to
  128k output tokens and the solution re-extracted from that prefix (Appendix B).
- `<name>.regraded.json` — re-scored on the fixed evaluator (Appendix C). Each record keeps
  `passed_before_regrade` beside the corrected `passed`, and the file keeps
  `pass@1_before_regrade`, so the correction is auditable rather than silent.

**Every figure and table in the paper reads the `.regraded.json` files**, and for
Qwen3.8-27B's single arm the `.cap128k.regraded.json` twin.

`q9-reasoning-on-archived/` is the evidence for §6.1's claim that Qwen3.5-9B with reasoning
enabled was untestable: these attempts were archived rather than graded, and the paper
reports the model as a limitation rather than a data point.

---

## 4. Reproducing

**The figures.** With `uv` available, from
`codebase/v2-current/escalation/run_bench_script/`:

```bash
uv run --with matplotlib --with numpy --with scipy python plot_agent_loop_flowchart.py   # Figure 1
uv run --with matplotlib --with numpy --with scipy python plot_4new_5pass_reason_on.py   # Figure 2
uv run --with matplotlib --with numpy --with scipy python plot_cost_5_pass.py            # Figure 3 + Table 1
uv run --with matplotlib --with numpy --with scipy python plot_cost_vs_score.py          # Figure 4
uv run --with matplotlib --with numpy --with scipy python plot_bars.py                   # Figures 5a-5c
uv run --with matplotlib --with numpy --with scipy python plot_128k_reason_on_1_pass.py  # Figure 6
uv run --with matplotlib --with numpy --with scipy python plot_16k_reason_off_5_pass.py  # Figure 7
uv run --with matplotlib --with numpy --with scipy python plot_128k_reason_off_1_pass.py # Figure 8
uv run --with matplotlib --with numpy --with scipy python make_figures_tex.py            # -> paper/fig-*.tex
```

Those nine commands write exactly the ten files `paper/plots/` holds and nothing else —
one vector PDF per figure, light theme only. Each prints its table to stdout first; those
printed numbers are the ones in the paper.

`plot_q38_vs_fable5_5_pass.py` still runs and is still where the manager-vs-Fable-5
numbers are computed, but the paper does not include its chart — Figure 2 carries the same
comparison — so it is not in the list above and a default run writes nothing for it.
`plot_agent_loop_flowchart.py` likewise keeps `FIG_46710A5`, the original scaffold's
diagram, defined but not emitted: it is the only drawing of the version behind §5.4.

**The grading.** To repeat the Appendix C correction from the raw generations:

```bash
python escalation/regrade.py runs/<condition>/results/*.json
```

**The runs themselves.** Both scaffolds are driven through `escalation/run_bench.py`,
which expects the benchmark harness importable from the repository root:

```bash
python escalation/run_bench.py --engine {single,multiagent} --only lcb --lcb 100 \
       --ids-file escalation/lcb100_hardest_v6.json --parallel N --out results.json
```

`run_bench_script/run_4models_1pass_reason_on.sh` is the driver for the three first-party
API arms and `run_fable5_5pass_single.sh` for Fable 5, both including the per-provider
caveats. Model routing, caps and reasoning mode are environment-driven; see the header
comments in `orchestrator.py`, and Appendix B of the paper for what each provider's thinking
control was set to.

---

## 5. What was deliberately left out

- **Credentials**: `.env*`, `*.key`, `.credentials*` — see the note at the top.
- **`.gitignore`**, per request.
- **One-off scripts not part of the benchmark**: `aggregate_big.py` and `run_big.sh`, which
  serve the exploratory LCB+AIME sweep that §6.1 lists among the benchmarks run but not
  reported.
- **Superseded reruns** inside the first-party sweep — the `.clampbug`, `.pre-notesfix` and
  `.pre-planfix` workspaces. The five graded passes per arm are what ship.
- **`LiveCodeBench/output/` and `LiveCodeBench/claude_transcripts/`** (~177 MB). These
  belong to a *separate* experiment in the same repository — a wrapper that benchmarks the
  `claude -p` CLI as an agent — which shares no data with this paper.
- **Dark-theme chart variants**, reproducible from the plot scripts. The light variants the
  paper uses ship in `paper/plots/`.
