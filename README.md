Estimate a randomized controlled trial's treatment effect, controlling for each student's prior knowledge.

The prior-knowledge control comes from knowledge-tracing models (BKT, DKT, DKT-P) trained on a **remnant**: problem
logs from students who were not in the trial. The trained models predict how each trial student will do on their
next problem, using only logs from before that problem. That prediction is then the control in
`outcome ~ treatment * control`. A control that predicts the outcome better gives a narrower confidence interval for
the treatment effect. The original control, Prior5 (the average of the last five scores), is included as the
baseline.

```bash
uv run python analyzeRCT.py <remnant> <columns> <rct_students> <rct_data> [model flags]
```

---

## Contents

- [Setup](#setup)
- [Quick start](#quick-start)
- [Inputs](#inputs)
- [schema.yaml](#schemayaml)
- [Models](#models)
- [Outputs](#outputs)
- [Reading the results](#reading-the-results)
- [Reusing trained models](#reusing-trained-models)
- [Optional: mixed models in R](#optional-mixed-models-in-r)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The CSVs in `Data/` are stored with Git LFS, so run
`git lfs pull` after cloning.

```bash
uv sync --no-install-package pybkt && uv sync
```

pyBKT is built from source to get its fast C++ fitter, so the first sync has to skip it. On macOS that build needs
OpenMP:

```bash
brew install libomp
uv sync --no-install-package pybkt
CPLUS_INCLUDE_PATH=$(brew --prefix libomp)/include LIBRARY_PATH=$(brew --prefix libomp)/lib uv sync
```

A CUDA GPU is used automatically if present. Otherwise training runs on CPU.

---

## Quick start

With the data in this repository:

```bash
uv run python analyzeRCT.py Data/Remnant.csv "end_time,discrete_score,problem_id,user_xid" \
    Data/PriorRCT.csv Data/RCT.csv --Prior5 --BKT --DKT --DKT-P --out results/all_models
```

On a laptop CPU, the first run takes about 70 minutes: about 10 for BKT, and the rest DKT and DKT-P training. Running again with the same
`--out` reuses the trained models and finishes in a few minutes. The report is written to
`results/all_models/summary.md`.

---

## Inputs

| # | Argument | What it is |
| --- | --- | --- |
| 1 | `remnant` | Problem logs from students **not** in the trial. The models are trained on these. |
| 2 | `columns` | The remnant's column names, as `order,score,skill/problem,user` or `order,score,problem,user`. |
| 3 | `rct_students` | Every problem log of the students **in** the trial, before and after it. |
| 4 | `rct_data` | The trial itself: one row per trial event, with the condition and the outcome. |

| Flag | What it does |
| --- | --- |
| `--Prior5` | Adds Prior5, the average of the student's last 5 scores (the original control). |
| `--BKT` | Adds Bayesian knowledge tracing. |
| `--DKT` | Adds deep knowledge tracing. |
| `--DKT-P` | Adds deep knowledge tracing with problem difficulty. |
| `--IRT` | Adds item response theory. Not implemented yet, see [Known limitations](#known-limitations). |
| `--schema PATH` | Column schema to use. Default: `schema.yaml` next to `rct_data`. |
| `--out DIR` | Where results go. Default: `results/<date-time>`. |

At least one model flag is required.

### The remnant's columns

- **order** is the order of a student's logs. It can be a log id (`id`) or a timestamp (`end_time`). Timestamps are
  parsed as ISO 8601.
- **score** is 0/1 correctness. Rows with any other value are dropped.
- **skill** is optional. Without it (`order,score,problem,user`), each problem is treated as its own skill. This is
  how the data in `Data/` is run, because `Remnant.csv` has no skill column.
- **problem** is the problem id. DKT-P uses it for per-problem difficulty, and it links rct_students to the
  remnant's skills.
- **user** is the student id.

Examples:

```text
id,discrete_score,skill_id/problem_id,user_xid      log id + skill
end_time,discrete_score,problem_id,user_xid         timestamp, no skill
```

### Data in this repository

| File | Role | Contents |
| --- | --- | --- |
| `Data/Remnant.csv` | remnant | 10.6M logs from 98,891 students, 2024-25 |
| `Data/PriorRCT.csv` | rct_students | 6.2M scored logs from the 20,865 trial students |
| `Data/RCT.csv` | rct_data | 88,726 trial events from 20,964 students, 49.4% treated |
| `Data/schema.yaml` | schema | Column mapping for the three files above |

---

## schema.yaml

The remnant's columns come from the command line. The columns of `rct_students` and `rct_data` are recorded in
`schema.yaml`.

1. **First run.** If there is no `schema.yaml`, the tool works out the columns (from their types, their names and
   which values match across the files), writes `schema.yaml`, and prints it.
2. **Check it.** Any column it couldn't decide is left blank and marked `UNSURE`, and the run stops until you fill
   those in. Check the guessed ones too.
3. **Later runs** read `schema.yaml` and skip the guessing.

`treatment`, `outcome` and each `exclude` rule can be a pandas expression instead of a column name.

```yaml
rct_students:
  order:     id
  score:     discrete_score
  problem:   problem_id
  user:      user_xid
rct_data:
  user:      user_xid
  log:       pl_id                        # matches rct_students.order
  problem:   problem_id                   # optional: clusters standard errors by problem
  class:     class_id                     # optional: clusters standard errors by class
  treatment: treatment                    # column or expression, 1 = treated
  outcome:   next_problem_across          # column or expression, 0/1
  prior5:    prior_5_across               # optional: blank = computed from rct_students
exclude:
  - user_xid <= 1000                      # internal test accounts
```

`rct_data.log` has to match `rct_students.order`. Each trial event's prediction target is the student's next scored
log after that event.

---

## Models

| Flag | Model | Trained on | Notes |
| --- | --- | --- | --- |
| `--Prior5` | Mean of the previous 5 scores | Nothing | Read from `rct_data.prior5` when set, otherwise computed from rct_students. |
| `--BKT` | Bayesian knowledge tracing with forgetting (pyBKT) | Remnant | One parameter set per skill, 5 EM restarts. Fitted in a subprocess (pyBKT's OpenMP clashes with PyTorch on macOS). |
| `--DKT` | LSTM over (skill, correct) | Remnant | Embed 100, hidden 200, lr 3e-3. |
| `--DKT-P` | LSTM over (skill, correct) + (problem, correct), with a next-problem embedding | Remnant | Embed 64, hidden 256, dropout 0.2, lr 1e-3. Problems seen fewer than 10 times share one embedding. |

DKT and DKT-P are trained for up to 30 epochs. Training stops early once validation AUC hasn't improved for 3 epochs,
using 10% of remnant students held out for validation. The hyperparameters are the winners of the grid search run
on Turing.

Predictions use only logs from before the target problem, so nothing after treatment leaks into the control.

---

## Outputs

```text
results/<name>/
├── schema.yaml        the columns used
├── predictions.csv    each trial event with its target log and every model's prediction
├── effects.csv        treatment effect per control and standard-error type
├── summary.md         the report to read
└── models/            trained models, reused by later runs with the same --out
    ├── bkt_params.csv
    └── dkt_models.pt
```

`results/` is ignored by git.

---

## Reading the results

The main table in `summary.md` has one row per control, plus **Unadjusted** (no control). All rows use the same
trial events: those every selected model could score.

| Column | Meaning |
| --- | --- |
| effect (pp) | Average treatment effect in percentage points: the mean difference between predicting every event as treated and as control. Shown as estimate ± 95% CI half-width. |
| odds ratio | Treatment odds ratio at the average value of the control, with its 95% CI. |
| AUC, R² | How well the control predicts the outcome. Higher values give narrower CIs. |

**Standard errors.** `effects.csv` has every type, and `summary.md` compares their CI widths.

| SE type | Accounts for |
| --- | --- |
| `model` | Nothing beyond the logistic model (events treated as independent). |
| `hc0` | Misspecified variance (White's robust SE). Events still treated as independent. |
| `student`, `class`, `problem` | Correlated events within one grouping. |
| `student+problem` | Correlation within students and within problems at once (two-way clustered). |

The main table uses `student+problem` when `rct_data` has a problem column, and `student` otherwise.

**Scales.** The percentage-point CI and the odds-ratio CI measure the same effect on different scales. Say which
one you are reporting. For example, ±0.63 pp and an odds-ratio CI 0.06 wide describe the same result.

---

## Reusing trained models

Training is the slow part. A model is trained only if `<out>/models` doesn't have it yet, so:

- Pass the same `--out` to reuse models from an earlier run, for example to add `--Prior5` or to rerun after editing
  `schema.yaml`.
- Copy a `bkt_params.csv` or `dkt_models.pt` from elsewhere (such as a Turing run) into `<out>/models` before
  running. The formats are the same as the Turing notebooks'.

If the remnant changes, use a new `--out`. Otherwise the old models are reused without warning.

---

## Optional: mixed models in R

`R/glmer.R` fits the mixed-model version of the analysis on a finished run's `predictions.csv`:
`glmer(outcome ~ treatment * control + (1|user) + (1|problem) + (1|class))`.

```bash
Rscript R/glmer.R results/all_models/predictions.csv results/all_models/glmer prior5 bkt dkt dkt_p
```

It writes odds ratios (`_terms.csv`), fit statistics (`_fit.csv`) and percentage-point effects (`_pp.csv`). It needs
`install.packages(c("lme4", "marginaleffects"))`.

---

## Tests

```bash
uv run python -m pytest -q
```

The tests run the pipeline end to end on small synthetic data and finish in about 10 seconds.

---

## Project layout

```text
analyzeRCT.py            the command you run
analyzerct/
├── columns.py           parse the column argument, infer columns, read and write schema.yaml
├── load.py              read the three files into one shape, apply exclude rules
├── predict.py           find each trial event's target log and score it with every model
├── effects.py           treatment effect, CIs under every standard-error type
└── models/
    ├── prior5.py
    ├── bkt.py
    ├── dkt.py           DKT and DKT-P
    └── irt.py           placeholder
R/glmer.R                optional mixed-model analysis
tests/                   synthetic end-to-end tests
Data/                    the MQP trial data (Git LFS)
cli-design/README.md     original design notes
```

---

## Known limitations

- **IRT is not implemented.** `irt.py` is a placeholder, so `--IRT` fails, and so does the test
  `test_every_model_end_to_end` (which uses it).
- **Problems as skills.** Without a skill column, BKT becomes a per-problem model and DKT has about 99,000 "skills".
  This works, but real skill ids would likely give stronger BKT and DKT models.
- **Schema inference needs unique log ids.** If `rct_students` repeats a log id (as `PriorRCT.csv` does), the order
  column is left blank for you to fill in.
- **Duplicate timestamps.** With a timestamp order column, two logs from the same student at the same millisecond
  count as one.
- **Events without a prediction are dropped.** Trial events with no later scored log, or whose problem is unknown
  to a model, are left out of every row. In the current data that is 85,877 of 88,726 events kept.
