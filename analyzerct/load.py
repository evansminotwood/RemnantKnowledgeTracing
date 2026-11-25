"""Read the three files and put them in one shape.

remnant / history: order_id, user_id, (skill_id,) problem_id, correct — scored (0/1) logs only, sorted by
(user, order). rct: user, log, problem, class, treatment, outcome, prior5 — one row per trial event.
"""
import numpy as np
import pandas as pd

from . import log
from .columns import referenced


def read(path, columns=None, nrows=None):
    if columns is not None:
        header = pd.read_csv(path, nrows=0).columns
        columns = [c for c in header if c in set(columns)]
    return pd.read_csv(path, usecols=columns, nrows=nrows)


def needed(fields, exclude, path):
    """Columns to read from a file: its schema columns plus any column an exclude rule mentions."""
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in fields.values() if c] + [c for e in exclude for c in referenced(e, header)]


def apply_exclude(df, exclude, name):
    for rule in exclude:
        try:
            drop = df.eval(rule, engine="python").fillna(False).astype(bool)
        except pd.errors.UndefinedVariableError:
            continue   # the rule is about a column this file doesn't have
        if drop.any():
            log(f"[{name}] exclude {rule!r}: {drop.sum():,} rows")
        df = df[~drop.to_numpy()]
    return df


def _scored_logs(df, cols, name):
    keep = {cols["order"]: "order_id", cols["user"]: "user_id", cols["problem"]: "problem_id", cols["score"]: "correct"}
    if cols.get("skill"):
        keep[cols["skill"]] = "skill_id"
    n = len(df)
    d = df[list(keep)].rename(columns=keep)
    unique_by = ["order_id"]   # a log id is unique across students
    if not pd.api.types.is_numeric_dtype(d["order_id"]):   # timestamps, e.g. end_time: unique per student only
        d["order_id"] = pd.to_datetime(d["order_id"], utc=True, format="ISO8601").astype("int64")
        unique_by = ["user_id", "order_id"]
    d = d.drop_duplicates(unique_by)
    d = d[d["correct"].isin([0, 1])].astype({"correct": int})
    d = d.sort_values(["user_id", "order_id"]).reset_index(drop=True)
    log(f"[{name}] {n:,} rows -> {len(d):,} scored logs | {d['user_id'].nunique():,} students | "
        f"{d['problem_id'].nunique():,} problems | mean correct {d['correct'].mean():.3f}")
    return d


def remnant(path, cols, exclude):
    df = apply_exclude(read(path, needed(cols, exclude, path)), exclude, "remnant")
    d = _scored_logs(df, cols, "remnant")
    if not cols.get("skill"):
        d["skill_id"] = d["problem_id"]   # no skill column: each problem is its own skill
    return d


def history(df, cols, exclude, skills):
    """Scored logs of the RCT students; skill_id comes from the remnant's problem -> skill map (NaN if unknown)."""
    d = _scored_logs(apply_exclude(df, exclude, "rct_students"), cols, "rct_students")
    d["skill_id"] = d["problem_id"].map(skills)
    log(f"[rct_students] logs whose problem has a remnant skill: {d['skill_id'].notna().mean():.1%}")
    return d


def rct(df, cols, exclude):
    """One row per trial event. treatment and outcome may be expressions; missing outcomes count as 0 and
    missing prior5 / class as 0, as in the original analysis."""
    n = len(df)
    df = apply_exclude(df, exclude, "rct_data")
    out = pd.DataFrame({"user": df[cols["user"]], "log": df[cols["log"]]})
    for f in ["problem", "class"]:
        out[f] = df[cols[f]].fillna(0) if cols[f] else 0
    treat = df.eval(cols["treatment"], engine="python")
    if not set(pd.unique(treat.dropna())) <= {0, 1}:
        raise SystemExit(f"rct_data.treatment ({cols['treatment']}) must be 0/1 or True/False")
    out["treatment"] = treat.fillna(0).astype(int)
    y = np.floor(df.eval(cols["outcome"], engine="python").astype(float))
    if not set(y.dropna().unique()) <= {0, 1}:
        raise SystemExit(f"rct_data.outcome ({cols['outcome']}) must be 0/1 after rounding down")
    out["outcome"] = y.fillna(0).astype(int)
    if cols["prior5"]:
        out["prior5"] = df[cols["prior5"]].fillna(0).astype(float)
    out = out.reset_index(drop=True)
    log(f"[rct_data] {n:,} rows -> {len(out):,} trial events | {out['user'].nunique():,} students | "
        f"treated share {out['treatment'].mean():.3f} | missing outcome set to 0: {y.isna().sum():,}")
    return out


def skill_map(remnant):
    """problem_id -> skill_id from the remnant (rct_students carries no skills)."""
    m = remnant.dropna(subset=["skill_id"]).drop_duplicates("problem_id")
    return m.set_index("problem_id")["skill_id"].astype(int)
