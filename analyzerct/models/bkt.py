"""Bayesian knowledge tracing: fit with pyBKT (one parameter set per skill, with forgetting), predict with a
forward pass.

Parameters are saved as pyBKT's `params()` table (skill, param, class, value), the same bkt_params.csv the Turing
runs write, so those models load here unchanged.
"""
import numpy as np
import pandas as pd

from .. import log
from . import positions, take, with_skill

FILE = "bkt_params.csv"
FORGETS = True     # won the Turing CV (val AUC 0.684 vs 0.673)
NUM_FITS = 5       # EM restarts per skill
COLUMNS = {"order_id": "order_id", "user_id": "user_id", "correct": "correct", "skill_name": "skill_id"}


def fit(remnant, seed=1):
    """Fit in a separate Python process: on macOS, pyBKT's OpenMP runtime (Homebrew libomp) crashes when PyTorch's
    bundled libomp is already loaded in the same process."""
    import os
    import subprocess
    import sys
    import tempfile
    df = with_skill(remnant)
    log(f"[BKT] fitting {df['skill_id'].nunique():,} skills on {len(df):,} logs")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.pkl"), os.path.join(tmp, "out.csv")
        df[["order_id", "user_id", "skill_id", "correct"]].to_pickle(src)
        code = ("import sys, pandas as pd; sys.path.insert(0, sys.argv[1]); from analyzerct.models.bkt import _fit_here; "
                "_fit_here(pd.read_pickle(sys.argv[2]), int(sys.argv[3])).to_csv(sys.argv[4])")
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.run([sys.executable, "-c", code, root, src, str(seed), dst], check=True)
        return pd.read_csv(dst)


def _fit_here(df, seed):
    from pyBKT.models import Model
    model = Model(seed=seed, num_fits=NUM_FITS)
    # pyBKT converts the skill column to str in place, so give it a copy
    model.fit(data=df.copy(), defaults=COLUMNS, forgets=FORGETS)
    return model.params()


def save(params, path):
    params.to_csv(path, index="skill" not in params.columns)


def load(path):
    return pd.read_csv(path)


def _param_dicts(params):
    p = params.reset_index() if "skill" not in params.columns else params
    wide = p.pivot_table(index="skill", columns="param", values="value")
    if "forgets" not in wide:
        wide["forgets"] = 0.0
    wide.index = wide.index.astype(int)
    return {c: wide[c].to_dict() for c in ["prior", "learns", "guesses", "slips", "forgets"]}


def forward(df, params):
    """P(correct) for every row of df (sorted by user, order), from the student's earlier logs on that skill.

    Identical to pyBKT's predict: P(L)·(1-slip) + (1-P(L))·guess, then a Bayesian update on the observed answer
    and a learn / forget transition. NaN for skills the model was not fit on.
    """
    p = _param_dicts(params)
    prior, learn, guess, slip, forget = p["prior"], p["learns"], p["guesses"], p["slips"], p["forgets"]
    out = np.full(len(df), np.nan)
    known = {}
    users, skills, correct = df["user_id"].to_numpy(), df["skill_id"].to_numpy(), df["correct"].to_numpy()
    for i in range(len(df)):
        k = skills[i]
        if k not in prior:
            continue
        key = (users[i], k)
        p_l = known.get(key, prior[k])
        g, s = guess[k], slip[k]
        out[i] = p_l * (1 - s) + (1 - p_l) * g
        # max(): degenerate fits (e.g. slip = 0 with P(L) = 1) give an observed answer probability 0
        post = p_l * (1 - s) / max(out[i], 1e-12) if correct[i] else p_l * s / max(1 - out[i], 1e-12)
        known[key] = post * (1 - forget[k]) + (1 - post) * learn[k]
    return out


def predict(history, target_ids, params):
    kt = with_skill(history)
    return take(forward(kt, params), positions(kt, target_ids))
