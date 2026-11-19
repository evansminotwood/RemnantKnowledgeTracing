"""Find the columns of rct_students and rct_data, and read / write schema.yaml.

A column is picked when it passes the field's type / value check and it is the only one that does at the first
level that narrows it down: the same name as in the other files, then name patterns from most to least specific.
Anything still ambiguous is left blank for the user to fill in.
"""
import json
import re

import numpy as np
import pandas as pd
import yaml

REMNANT_FIELDS = ["order", "score", "skill", "problem", "user"]
FIELDS = {"rct_students": ["order", "score", "problem", "user"],
          "rct_data": ["user", "log", "problem", "class", "treatment", "outcome", "prior5"]}
OPTIONAL = {"rct_data": {"problem", "class", "prior5"}}
COMMENTS = {("rct_data", "log"): "matches rct_students.order",
            ("rct_data", "problem"): "optional: clusters standard errors by problem",
            ("rct_data", "class"): "optional: clusters standard errors by class",
            ("rct_data", "treatment"): "column or expression, 1 = treated",
            ("rct_data", "outcome"): "column or expression, 0/1",
            ("rct_data", "prior5"): "optional: blank = computed from rct_students"}


def parse_colspec(spec):
    """'id,discrete_score,skill_id/problem_id,user_xid' -> {order, score, skill, problem, user}."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4 or parts[2].count("/") != 1:
        raise ValueError(f"expected order,score,skill/problem,user, got {spec!r}")
    skill, problem = parts[2].split("/")
    return dict(zip(REMNANT_FIELDS, [parts[0], parts[1], skill, problem, parts[3]]))


# ---------------------------------------------------------------- checks

def _is_int(s):
    v = s.dropna()
    return pd.api.types.is_numeric_dtype(s) and len(v) > 0 and bool(np.all(np.mod(v, 1) == 0))


def _is_binary(s):
    v = s.dropna()
    # tolerate a few stray codes (PriorData has one discrete_score of -1); only 0/1 rows are used
    return pd.api.types.is_numeric_dtype(s) and len(v) > 0 and v.isin([0, 1]).mean() >= 0.999


def _in_unit(s):
    v = s.dropna()
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and len(v) > 0 \
        and v.min() >= 0 and v.max() <= 1


def _unique(s):
    return s.nunique() >= 0.99 * s.notna().sum()


def _share_in(s, values):
    u = pd.Series(s.dropna().unique())
    return len(u) > 0 and u.isin(values).mean()


def _pick(df, check, tiers, used):
    """The one column passing `check` at the first tier (exact names, then regexes) that has exactly one match."""
    passing = [c for c in df.columns if c not in used and check(df[c])]
    for tier in tiers:
        hits = [c for c in passing if any(c == t if not t.startswith("re:") else re.search(t[3:], c, re.I) for t in tier)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, f"could be any of {hits}"
    return None, f"could be any of {passing}" if passing else "no column fits"


def infer(remnant_cols, remnant, students, rct):
    """Guess the schema. remnant: sample of the remnant with its problem column; students / rct: the full files.
    Returns (schema, notes) with notes[(file, field)] explaining every blank."""
    r = remnant_cols
    s, notes = {}, {}
    remnant_problems = remnant[r["problem"]].dropna().unique()

    def pick(file, field, df, check, tiers):
        col, note = _pick(df, check, tiers, used)
        if col:
            used.add(col)
        else:
            notes[(file, field)] = note
        return col

    used = set()
    st = {"user": pick("rct_students", "user", students, _is_int, [[r["user"]], ["re:user|student"]]),
          "score": pick("rct_students", "score", students, _is_binary, [[r["score"]], ["re:score|correct"]])}
    st["problem"] = pick("rct_students", "problem", students,
                         lambda c: _is_int(c) and _share_in(c, remnant_problems) >= 0.5,
                         [[r["problem"]], ["re:problem|item|question"]])
    st["order"] = pick("rct_students", "order", students, lambda c: _is_int(c) and _unique(c),
                       [[r["order"]], ["re:^pl_id$|order|log"], ["re:id$"]])
    s["rct_students"] = {f: st[f] for f in FIELDS["rct_students"]}

    used = set()
    users = students[st["user"]].unique() if st["user"] else []
    logs = students[st["order"]].unique() if st["order"] else []
    probs = students[st["problem"]].unique() if st["problem"] else remnant_problems
    rd = {"user": pick("rct_data", "user", rct, lambda c: _is_int(c) and _share_in(c, users) >= 0.95,
                       [[st["user"], r["user"]], ["re:user|student"], ["re:."]]),
          "log": pick("rct_data", "log", rct, lambda c: _is_int(c) and _unique(c) and _share_in(c, logs) >= 0.95,
                      [[st["order"]], ["re:log|order"], ["re:."]]),
          "problem": pick("rct_data", "problem", rct, lambda c: _is_int(c) and _share_in(c, probs) >= 0.9,
                          [[st["problem"], r["problem"]], ["re:problem|item|question"], ["re:."]]),
          "class": pick("rct_data", "class", rct, _is_int, [["re:class|section"], ["re:teacher|course"]]),
          "treatment": pick("rct_data", "treatment", rct, _is_binary,
                            [["re:treat"], ["re:condition|arm|assign|group"]]),
          "outcome": pick("rct_data", "outcome", rct, _in_unit,
                          [["re:outcome"], ["re:next_problem"], ["re:next|post"]]),
          "prior5": pick("rct_data", "prior5", rct, _in_unit, [["re:prior_?5|last_?5"], ["re:prior"]])}
    s["rct_data"] = {f: rd[f] for f in FIELDS["rct_data"]}
    s["remnant"] = dict(remnant_cols)
    s["exclude"] = []
    return s, notes


def missing(schema):
    """Required fields that are still blank."""
    return [f"{file}.{f}" for file, fields in FIELDS.items() for f in fields
            if not schema.get(file, {}).get(f) and f not in OPTIONAL.get(file, set())]


# ---------------------------------------------------------------- schema.yaml

def write(schema, path, notes=None):
    notes = notes or {}
    lines = ["# schema.yaml: which column is which. Written by analyzeRCT.py; check it and edit by hand.",
             "# Blank = not found. treatment, outcome and exclude take pandas expressions, e.g. strategy != '[]'.",
             "remnant:   # from the command line"]
    lines += [f"  {f + ':':<10} {schema['remnant'][f]}" for f in REMNANT_FIELDS]
    for file, fields in FIELDS.items():
        lines.append(f"{file}:")
        for f in fields:
            v = schema[file].get(f)
            required = f not in OPTIONAL.get(file, set())
            comment = f"UNSURE: {notes[(file, f)]}" if v is None and required and (file, f) in notes \
                else COMMENTS.get((file, f), "")
            val = _yaml_scalar(v) if v is not None else ""
            lines.append(f"  {f + ':':<10} {val:<28}" + (f" # {comment}" if comment else "").rstrip())
    lines.append("exclude:   # rows to drop from every file that has the column, e.g. user_xid <= 1000")
    lines += [f"  - {_yaml_scalar(e)}" for e in schema.get("exclude", [])]
    path.write_text("\n".join(l.rstrip() for l in lines) + "\n")


def _yaml_scalar(v):
    """Plain if YAML reads it back unchanged (e.g. strategy != '[]'), else double-quoted."""
    try:
        if yaml.safe_load(f"k: {v}")["k"] == v:
            return v
    except yaml.YAMLError:
        pass
    return json.dumps(v)


def read(path):
    s = yaml.safe_load(path.read_text()) or {}
    for file, fields in FIELDS.items():
        s[file] = {f: (None if (s.get(file) or {}).get(f) in (None, "") else str(s[file][f])) for f in fields}
    s["exclude"] = [str(e) for e in (s.get("exclude") or [])]
    return s


def referenced(expr, header):
    """Columns of `header` that appear in an expression."""
    return [c for c in header if re.search(rf"(?<![\w`]){re.escape(c)}(?![\w`])|`{re.escape(c)}`", expr)]
