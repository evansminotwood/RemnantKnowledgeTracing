#!/usr/bin/env python
"""Estimate an RCT's treatment effect, controlling for prior knowledge with models trained on the remnant.

    python analyzeRCT.py <remnant> <order,score,skill/problem,user> <rct_students> <rct_data> [flags]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyzerct import columns, effects, load, log, predict
from analyzerct.models import bkt, dkt, irt

FLAGS = ["Prior5", "BKT", "DKT", "DKT-P", "IRT"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("remnant", help="problem logs from students not in the RCT; the models are trained on these")
    p.add_argument("columns", help="the remnant's column names: order,score,skill/problem,user")
    p.add_argument("rct_students", help="every problem log of the students in the RCT")
    p.add_argument("rct_data", help="the RCT itself: one row per trial event, with the condition and outcome")
    p.add_argument("--Prior5", action="store_true", help="average of the student's last 5 scores (the original control)")
    p.add_argument("--BKT", action="store_true", help="Bayesian knowledge tracing")
    p.add_argument("--DKT", action="store_true", help="deep knowledge tracing")
    p.add_argument("--DKT-P", dest="DKT_P", action="store_true", help="deep knowledge tracing with problem difficulty")
    p.add_argument("--IRT", action="store_true", help="item response theory")
    p.add_argument("--schema", type=Path, help="column schema to use (default: schema.yaml next to rct_data)")
    p.add_argument("--out", type=Path, help="where results go (default: results/<date-time>)")
    a = p.parse_args(argv)
    a.flags = [f for f in FLAGS if getattr(a, f.replace("-", "_"))]
    if not a.flags:
        p.error("pick at least one of --Prior5 --BKT --DKT --DKT-P --IRT")
    try:
        a.remnant_cols = columns.parse_colspec(a.columns)
    except ValueError as e:
        p.error(str(e))
    a.schema = a.schema or Path(a.rct_data).parent / "schema.yaml"
    a.out = a.out or Path("results") / time.strftime("%Y-%m-%d-%H%M%S")
    return a


def get_schema(a):
    """Read schema.yaml, or find the columns and write it. Returns (schema, rct_students frame, rct_data frame)."""
    if a.schema.exists():
        s = columns.read(a.schema)
        s["remnant"] = a.remnant_cols
        students = load.read(a.rct_students, load.needed(s["rct_students"], s["exclude"], a.rct_students))
        rct = load.read(a.rct_data)
        log(f"[schema] using {a.schema}")
    else:
        students, rct = load.read(a.rct_students), load.read(a.rct_data)
        sample = load.read(a.remnant, [a.remnant_cols["problem"]], nrows=1_000_000)
        s, notes = columns.infer(a.remnant_cols, sample, students, rct)
        columns.write(s, a.schema, notes)
        log(f"[schema] wrote {a.schema}:\n\n{a.schema.read_text()}")
    todo = columns.missing(s)
    if todo:
        sys.exit(f"[schema] fill in {', '.join(todo)} in {a.schema}, then run again")
    for file, df in [("rct_students", students), ("rct_data", rct)]:
        absent = [c for f, c in s[file].items() if c and f not in ("treatment", "outcome") and c not in df]
        if absent:
            sys.exit(f"[schema] {file} has no column {absent}; fix {a.schema}")
    return s, students, rct


def get_models(a, remnant, device):
    """Load each model from <out>/models, or train it on the remnant and save it there."""
    mdir = a.out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    models = {}
    if "BKT" in a.flags:
        path = mdir / bkt.FILE
        if not path.exists():
            bkt.save(bkt.fit(remnant), path)
        models["bkt"] = bkt.load(path)
    names = [n for n in ["DKT", "DKT-P"] if n in a.flags]
    if names:
        path = mdir / dkt.FILE
        ckpt = dkt.load(path, device) if path.exists() else None
        todo = [n for n in names if ckpt is None or n not in ckpt[0]]
        if todo:
            ckpt = dkt.fit(remnant, todo, device, ckpt)
            dkt.save(ckpt, path)
        models["dkt"] = ckpt
    if "IRT" in a.flags:
        path = mdir / irt.FILE
        if not path.exists():
            irt.save(irt.fit(remnant, device=str(device)), path)
        models["irt"] = irt.load(path)
    log(f"[models] {', '.join(a.flags)} ready in {mdir}")
    return models


def estimate(pred, flags, se_types):
    """Treatment effect without a control and with each model's prediction as the control, on the trial events
    every model could score."""
    cols = [predict.COLUMN[f] for f in flags]
    d = pred.dropna(subset=cols)
    y, t = d["outcome"].to_numpy().astype(float), d["treatment"].to_numpy()
    clusters = {"student": d["user"].to_numpy(), "class": d["class"].to_numpy(), "problem": d["problem"].to_numpy()}
    rows = []
    for name in ["Unadjusted"] + flags:
        x = None if name == "Unadjusted" else d[predict.COLUMN[name]].to_numpy(dtype=float)
        res, fit = effects.analyze(y, t, x, clusters, se_types=se_types)
        rows.append(res.assign(control=name, **fit))
        log(f"[effects] {name} done")
    eff = pd.concat(rows, ignore_index=True)
    eff = eff[["control"] + [c for c in eff.columns if c != "control"]]
    return eff, effects.raw_difference(y, t), d


def summary_md(a, eff, raw, d, pred, se):
    s = eff[eff["se_type"] == se]
    lines = [
        "# analyzeRCT results", "",
        f"- remnant: `{a.remnant}`", f"- rct_students: `{a.rct_students}`", f"- rct_data: `{a.rct_data}`",
        f"- trial events: {len(d):,} of {len(pred):,} scored by every model | {d['user'].nunique():,} students | "
        f"treated share {d['treatment'].mean():.3f}", "",
        f"## Treatment effect, percentage points (95% CI, SE clustered by {se.replace('+', ' and ')})", "",
        "| control | effect (pp) | 95% CI | p | odds ratio [95% CI] | AUC | R² |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in s.iterrows():
        lines.append(f"| {r['control']} | {r['effect (pp)']:+.2f} ± {r['pp ± (half-width)']:.2f} | "
                     f"[{r['pp lo']:+.2f}, {r['pp hi']:+.2f}] | {r['pp p']:.3f} | "
                     f"{r['treatment OR']:.3f} [{r['OR lo']:.3f}, {r['OR hi']:.3f}] | "
                     f"{r['AUC (fixed effects)']:.3f} | {r['latent R2']:.3f} |")
    lines += [f"| raw difference | {raw['effect (pp)']:+.2f} ± {raw['pp ± (half-width)']:.2f} | "
              f"[{raw['pp lo']:+.2f}, {raw['pp hi']:+.2f}] | | | | |", "",
              "Model: `outcome ~ treatment * control` (logistic, control centered). The effect is the average "
              "difference between predicting every event as treated and as control; ± is the half-width of the 95% CI. "
              "AUC and R² (latent scale) say how well the model predicts the outcome. "
              "The raw difference uses the textbook SE, which treats events as independent.", "",
              "## 95% CI half-width (±, percentage points) by standard error", ""]
    w = eff.pivot_table(index="control", columns="se_type", values="pp ± (half-width)", sort=False)
    w = w[[c for c in effects.SE_TYPES if c in w]]
    lines += ["| control | " + " | ".join(w.columns) + " |", "| --- |" + " ---: |" * len(w.columns)]
    lines += [f"| {i} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for i, r in w.iterrows()]
    lines += ["", "`model`: independent events · `hc0`: robust · `student` / `class` / `problem`: clustered · "
              "`student+problem`: two-way clustered. Every SE type is in effects.csv."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    a = parse_args(argv)
    schema, students_raw, rct_raw = get_schema(a)
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.schema, a.out / "schema.yaml")
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[run] results -> {a.out} | device {device}")

    remnant = load.remnant(a.remnant, a.remnant_cols, schema["exclude"])
    history = load.history(students_raw, schema["rct_students"], schema["exclude"], load.skill_map(remnant))
    rct = load.rct(rct_raw, schema["rct_data"], schema["exclude"])
    del students_raw, rct_raw

    models = get_models(a, remnant, device)
    pred = predict.predict(rct, history, remnant, models, a.flags, device)
    pred.to_csv(a.out / "predictions.csv", index=False)

    se_types = ["model", "hc0", "student"] + (["class"] if schema["rct_data"]["class"] else []) + \
               (["problem", "student+problem"] if schema["rct_data"]["problem"] else [])
    eff, raw, d = estimate(pred, a.flags, se_types)
    eff.to_csv(a.out / "effects.csv", index=False)
    summary = summary_md(a, eff, raw, d, pred, se_types[-1] if schema["rct_data"]["problem"] else "student")
    (a.out / "summary.md").write_text(summary)
    log("\n" + summary + f"\n[run] done -> {a.out}")


if __name__ == "__main__":
    main()
