"""Predict each RCT student's next problem from their earlier logs.

For every trial event, the target is the student's next scored log in rct_students after the trial log; each
model predicts it from the logs before it only, so nothing after treatment leaks in.
"""
import numpy as np
import pandas as pd

from . import log
from .models import bkt, dkt, irt, prior5

COLUMN = {"Prior5": "prior5", "BKT": "bkt", "DKT": "dkt", "DKT-P": "dkt_p", "IRT": "irt"}


def next_scored_log(rct, history):
    """order_id of each event's next scored log (NaN when the student has none)."""
    left = rct[["user", "log"]].rename(columns={"user": "user_id"}).reset_index()
    left["key"] = left["log"] + 1   # strictly after the trial log
    right = history[["user_id", "order_id"]].sort_values("order_id")
    m = pd.merge_asof(left.sort_values("key"), right, left_on="key", right_on="order_id", by="user_id",
                      direction="forward").set_index("index").sort_index()
    log(f"[predict] {m['order_id'].notna().sum():,} of {len(rct):,} trial events have a later scored log")
    return m["order_id"].to_numpy()


def predict(rct, history, remnant, models, flags, device):
    """rct plus target_log and one prediction column per flag."""
    out = rct.copy()
    out["target_log"] = next_scored_log(rct, history)
    targets = out["target_log"].to_numpy()
    if "Prior5" in flags and "prior5" not in rct:
        out["prior5"] = prior5.predict(history, targets, fill=remnant["correct"].mean())
    if "BKT" in flags:
        out["bkt"] = bkt.predict(history, targets, models["bkt"])
    names = [n for n in ["DKT", "DKT-P"] if n in flags]
    if names:
        for name, pr in dkt.predict(history, targets, models["dkt"], names, device).items():
            out[COLUMN[name]] = pr
    if "IRT" in flags:
        out["irt"] = irt.predict(history, targets, models["irt"])
    for f in flags:
        log(f"[predict] {f}: {out[COLUMN[f]].notna().sum():,} events scored | mean {out[COLUMN[f]].mean():.3f}")
    return out
