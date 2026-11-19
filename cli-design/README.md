# analyzeRCT

Estimate an RCT's treatment effect, controlling for prior knowledge with models trained on the remnant.

```bash
python analyzeRCT.py <remnant> <order,score,skill/problem,user> <rct_students> <rct_data> [flags]
```

```bash
python analyzeRCT.py data/remnant_2425.csv "id,discrete_score,skill_id/problem_id,user_xid" \
    data/rct_students.csv data/rct_data.csv --BKT --DKT --DKT-P --IRT --Prior5
```

---

## Arguments

| # | Argument | What it is |
| --- | --- | --- |
| 1 | `remnant` | path to problem logs from students **not** in the RCT; the models are trained on these |
| 2 | `order,score,skill/problem,user` | the remnant's column names, in this order |
| 3 | `rct_students` | path to every problem log of the students **in** the RCT |
| 4 | `rct_data` | path to the RCT itself: one row per trial event, with the condition and outcome |

| Flag | Adds |
| --- | --- |
| `--Prior5` | average of the student's last 5 scores (the original control) |
| `--BKT` | Bayesian knowledge tracing |
| `--DKT` | deep knowledge tracing |
| `--DKT-P` | deep knowledge tracing with problem difficulty |
| `--IRT` | item response theory |
| `--schema PATH` | column schema to use (default: `schema.yaml` next to `rct_data`) |
| `--out DIR` | where results go (default: `results/<date-time>`) |

---

## Columns: first run writes a schema, you adjust it

```mermaid
flowchart LR
    A[first run] --> B[finds the columns in rct_students and rct_data]
    B --> C{{writes schema.yaml}}
    C --> D[you check and edit it]
    D --> E[next run uses it]
```

1. **First run.** The remnant's columns come from argument 2. For the other two files the tool finds the
   columns itself (by type, name and which values match across files), writes them to `schema.yaml`, and prints
   what it found.
2. **Adjust.** Anything it got wrong or wasn't sure about, fix in `schema.yaml`. Unsure columns are left blank
   and the run stops until they are filled in.
3. **Every later run** reads `schema.yaml` and skips the guessing.

```yaml
# schema.yaml (written by the first run, edited by hand)
remnant:
  order:   id
  score:   discrete_score
  skill:   skill_id
  problem: problem_id
  user:    user_xid
rct_students:
  order:   pl_id
  score:   discrete_score
  problem: problem_id
  user:    user_xid
rct_data:
  user:      user_xid
  log:       pl_id                      # matches rct_students.order
  problem:   problem_id
  class:     class_id
  treatment: selected_tutor_strategy_ceris != '[]'
  outcome:   next_problem_across
  prior5:    prior_5_across
exclude:
  - user_xid <= 1000                    # internal test accounts
```

---

## Project layout

```
analyzeRCT/
├── analyzeRCT.py        # the command you run
├── analyzerct/          # the code behind it
│   ├── columns.py       #   find columns, read and write schema.yaml
│   ├── load.py          #   read the three files
│   ├── models/          #   one file per flag: prior5, bkt, dkt (DKT + DKT-P), irt
│   ├── predict.py       #   predict each RCT student's next problem from their earlier logs
│   └── effects.py       #   treatment effect and confidence intervals
├── R/                   # optional R versions of the analysis
├── tests/
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
uv sync --no-install-package pybkt && uv sync      # pyBKT is built from source for its fast C++ fitter
uv run pytest                                       # small synthetic end-to-end check
```

On macOS the pyBKT build needs OpenMP: `brew install libomp`, then run the second `uv sync` as
`CPLUS_INCLUDE_PATH=$(brew --prefix libomp)/include LIBRARY_PATH=$(brew --prefix libomp)/lib uv sync`.

R (`lme4`, `marginaleffects`) is only needed for the optional mixed models, run on a finished result:

```bash
Rscript R/glmer.R results/<date-time>/predictions.csv results/<date-time>/glmer prior5 bkt dkt dkt_p irt
```

---

## Outputs

```
results/<date-time>/
├── schema.yaml          # the columns used
├── predictions.csv      # each RCT event with every model's prediction
├── effects.csv          # treatment effect per model, with confidence intervals
├── summary.md           # the report to read
└── models/              # trained models, reused by later runs
```

Training is the slow part (DKT especially). A model is trained only when `<out>/models` doesn't have it yet, so
pass the same `--out` again to reuse them, or copy trained `bkt_params.csv` / `dkt_models.pt` / `irt_params.csv`
(e.g. from a Turing run) into `<out>/models` first.
