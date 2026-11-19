"""One module per flag. Each trained model predicts P(correct) for target logs of a history frame from the
student's earlier logs only, returning one value per target id (NaN where it can't score that log)."""
import numpy as np
import pandas as pd


def positions(frame, target_ids):
    """Row of `frame` holding each target order_id; -1 where the target is missing or not in frame."""
    t = pd.Series(target_ids)
    out = np.full(len(t), -1)
    ok = t.notna().to_numpy()
    pos = pd.Series(np.arange(len(frame)), index=frame["order_id"].to_numpy())
    out[ok] = pos.reindex(t[ok].astype("int64").to_numpy()).fillna(-1).astype(int).to_numpy()
    return out


def take(values, pos):
    out = np.full(len(pos), np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def with_skill(history):
    return history.dropna(subset=["skill_id"]).astype({"skill_id": int}).reset_index(drop=True)
