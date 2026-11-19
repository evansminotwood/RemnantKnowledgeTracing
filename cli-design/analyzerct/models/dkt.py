"""Deep knowledge tracing (DKT) and DKT with problem difficulty (DKT-P), both kept in one checkpoint.

Model classes, tensor layout and checkpoint format are the ones in bkt_dkt_2425.ipynb, so the dkt_models.pt from
the Turing runs loads here unchanged. Hyperparameters are the Turing grid-search winners.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .. import log
from . import positions, with_skill

FILE = "dkt_models.pt"
PARAMS = {"DKT": {"embed_dim": 100, "hidden_dim": 200, "lr": 3e-3, "weight_decay": 0.0},
          "DKT-P": {"embed_dim": 64, "hidden_dim": 256, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5}}
MAX_LEN = 200            # window length (training windows and prediction context)
EPOCHS = 30
PATIENCE = 3             # early stopping: epochs without a validation AUC gain
BATCH_SIZE = 256
VAL_FRAC = 0.1           # remnant students held out for early stopping
MIN_PROBLEM_COUNT = 10   # rarer problems share one "unknown problem" embedding


class DKTModel(nn.Module):
    """LSTM over (skill, correct) embeddings, one output per skill; the next problem's skill output is used."""

    def __init__(self, num_skills, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(2 * num_skills + 1, embed_dim, padding_idx=2 * num_skills)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_skills)

    def forward(self, b):
        h = self.lstm(self.embedding(b["in_skill"]))[0]
        return self.fc(h).gather(-1, b["tgt_skill"].unsqueeze(-1)).squeeze(-1)


class ProblemDKTModel(nn.Module):
    """LSTM over (skill, correct) + (problem, correct) embeddings; the hidden state is combined with an embedding of
    the next problem's skill + problem, so the model can learn per-problem difficulty."""

    def __init__(self, num_skills, num_problems, embed_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.in_skill = nn.Embedding(2 * num_skills + 1, embed_dim, padding_idx=2 * num_skills)
        self.in_problem = nn.Embedding(2 * num_problems + 1, embed_dim, padding_idx=2 * num_problems)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.q_skill = nn.Embedding(num_skills, embed_dim)
        self.q_problem = nn.Embedding(num_problems, embed_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim + embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, b):
        h = self.lstm(self.in_skill(b["in_skill"]) + self.in_problem(b["in_problem"]))[0]
        q = self.q_skill(b["tgt_skill"]) + self.q_problem(b["tgt_problem"])
        return self.out(torch.cat([h, q], dim=-1)).squeeze(-1)


def make_model(name, params, num_skills, num_problems):
    if name == "DKT":
        return DKTModel(num_skills, params["embed_dim"], params["hidden_dim"])
    return ProblemDKTModel(num_skills, num_problems, params["embed_dim"], params["hidden_dim"], params["dropout"])


# ---------------------------------------------------------------- tensors

def training_tensors(df, skill_to_idx, problem_to_idx, max_len=MAX_LEN):
    """Student histories split into windows of max_len; position t holds interaction t as input and
    interaction t+1 as target. row_idx maps each target back to its row of df (-1 = padding)."""
    num_skills, num_problems = len(skill_to_idx), len(problem_to_idx) + 1
    skill_idx = df["skill_id"].map(skill_to_idx).to_numpy()
    problem_idx = df["problem_id"].map(problem_to_idx).fillna(0).astype(int).to_numpy()
    correct = df["correct"].to_numpy()
    users = df["user_id"].to_numpy()
    bounds = np.flatnonzero(users[1:] != users[:-1]) + 1
    windows = [(ws, min(ws + max_len, e)) for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(users)])
               for ws in range(s, e, max_len) if min(ws + max_len, e) - ws >= 2]
    shape = (len(windows), max_len - 1)
    t = {"in_skill": np.full(shape, 2 * num_skills, np.int64), "in_problem": np.full(shape, 2 * num_problems, np.int64),
         "tgt_skill": np.zeros(shape, np.int64), "tgt_problem": np.zeros(shape, np.int64),
         "tgt_correct": np.zeros(shape, np.float32), "row_idx": np.full(shape, -1, np.int64)}
    for r, (ws, we) in enumerate(windows):
        n = we - ws - 1
        t["in_skill"][r, :n] = skill_idx[ws:we - 1] * 2 + correct[ws:we - 1]
        t["in_problem"][r, :n] = problem_idx[ws:we - 1] * 2 + correct[ws:we - 1]
        t["tgt_skill"][r, :n] = skill_idx[ws + 1:we]
        t["tgt_problem"][r, :n] = problem_idx[ws + 1:we]
        t["tgt_correct"][r, :n] = correct[ws + 1:we]
        t["row_idx"][r, :n] = np.arange(ws + 1, we)
    return {k: torch.from_numpy(v) for k, v in t.items()}


def target_tensors(df, target_pos, skill_to_idx, problem_to_idx, max_len=MAX_LEN):
    """One window per target row: the student's previous max_len-1 rows of df as input, the target as the last
    label. `last` is the position of the target in its window (-1: no earlier row, so no prediction)."""
    num_skills, num_problems = len(skill_to_idx), len(problem_to_idx) + 1
    skill_idx = df["skill_id"].map(skill_to_idx).to_numpy()
    problem_idx = df["problem_id"].map(problem_to_idx).fillna(0).astype(int).to_numpy()
    correct = df["correct"].to_numpy()
    user_start = pd.Series(np.arange(len(df))).groupby(df["user_id"].to_numpy()).transform("min").to_numpy()
    shape = (len(target_pos), max_len - 1)
    t = {"in_skill": np.full(shape, 2 * num_skills, np.int64), "in_problem": np.full(shape, 2 * num_problems, np.int64),
         "tgt_skill": np.zeros(shape, np.int64), "tgt_problem": np.zeros(shape, np.int64)}
    last = np.zeros(len(target_pos), np.int64)
    for r, te in enumerate(target_pos):
        ws = max(user_start[te], te - (max_len - 1))
        n = te - ws
        t["in_skill"][r, :n] = skill_idx[ws:te] * 2 + correct[ws:te]
        t["in_problem"][r, :n] = problem_idx[ws:te] * 2 + correct[ws:te]
        t["tgt_skill"][r, :n] = skill_idx[ws + 1:te + 1]
        t["tgt_problem"][r, :n] = problem_idx[ws + 1:te + 1]
        last[r] = n - 1
    return {k: torch.from_numpy(v) for k, v in t.items()}, last


# ---------------------------------------------------------------- training / prediction

def _batch(tensors, idx, device):
    return {k: v[idx].to(device) for k, v in tensors.items() if k != "row_idx"}


def _predict_positions(model, tensors, device, batch_size=1024):
    """P(correct) at every window position, shape (n_windows, seq_len)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tensors["in_skill"]), batch_size):
            out.append(torch.sigmoid(model(_batch(tensors, slice(i, i + batch_size), device))).cpu())
    return torch.cat(out)


def _train_one(model, name, params, fit_t, val_t, device, seed=0):
    """Adam + BCE, early stopping on validation AUC; returns the best-epoch model."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 0))
    crit = nn.BCEWithLogitsLoss(reduction="none")
    n_fit = len(fit_t["row_idx"])
    val_mask = val_t["row_idx"] >= 0
    val_y = val_t["tgt_correct"][val_mask].numpy()
    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_fit)
        loss_sum = n_sum = 0.0
        for i in range(0, n_fit, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            b = _batch(fit_t, idx, device)
            m = (fit_t["row_idx"][idx] >= 0).float().to(device)
            opt.zero_grad()
            loss = (crit(model(b), b["tgt_correct"]) * m).sum() / m.sum()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * m.sum().item()
            n_sum += m.sum().item()
        val_auc = roc_auc_score(val_y, _predict_positions(model, val_t, device)[val_mask].numpy())
        log(f"[{name}] epoch {epoch}/{EPOCHS} loss {loss_sum / n_sum:.4f} | val AUC {val_auc:.4f} | {time.time() - t0:.0f}s")
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            log(f"[{name}] early stop: no val AUC improvement for {PATIENCE} epochs")
            break
    model.load_state_dict(best_state)
    log(f"[{name}] best epoch {best_epoch} (val AUC {best_auc:.4f})")
    return model.eval()


def fit(remnant, names, device, ckpt=None, seed=0, split_seed=42):
    """Train `names` on the remnant (VAL_FRAC of students held out for early stopping) and add them to ckpt,
    keeping its skill / problem vocabularies when it has them. ckpt = (models, params, skill_to_idx, problem_to_idx)."""
    kt = with_skill(remnant)
    users = kt["user_id"].unique()
    val_users = set(users[np.random.default_rng(split_seed).permutation(len(users))[:int(VAL_FRAC * len(users))]])
    is_val = kt["user_id"].isin(val_users)
    fit_df, val_df = kt[~is_val].reset_index(drop=True), kt[is_val].reset_index(drop=True)
    if ckpt is None:
        counts = fit_df["problem_id"].value_counts()
        vocab = {p: i + 1 for i, p in enumerate(sorted(counts[counts >= MIN_PROBLEM_COUNT].index))}
        ckpt = ({}, {}, {s: i for i, s in enumerate(sorted(kt["skill_id"].unique()))}, vocab)
    models, params, s2i, p2i = ckpt
    fit_df, val_df = fit_df[fit_df["skill_id"].isin(s2i)], val_df[val_df["skill_id"].isin(s2i)]
    log(f"[DKT] {len(fit_df):,} training / {len(val_df):,} validation logs | {len(s2i):,} skills | {len(p2i):,} problems")
    fit_t, val_t = training_tensors(fit_df, s2i, p2i), training_tensors(val_df, s2i, p2i)
    for name in names:
        model = make_model(name, PARAMS[name], len(s2i), len(p2i) + 1)
        models[name] = _train_one(model, name, PARAMS[name], fit_t, val_t, device, seed)
        params[name] = PARAMS[name]
    return models, params, s2i, p2i


def save(ckpt, path):
    models, params, s2i, p2i = ckpt
    torch.save({"state_dicts": {k: m.state_dict() for k, m in models.items()}, "params": params,
                "skill_to_idx": s2i, "problem_to_idx": p2i}, path)


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    s2i, p2i = ck["skill_to_idx"], ck["problem_to_idx"]
    models = {}
    for name, state in ck["state_dicts"].items():
        m = make_model(name, ck["params"][name], len(s2i), len(p2i) + 1)
        m.load_state_dict(state)
        models[name] = m.to(device).eval()
    return models, ck["params"], s2i, p2i


def predict(history, target_ids, ckpt, names, device, batch_size=2048):
    """{name: P(correct) per target id}."""
    models, _, s2i, p2i = ckpt
    d = with_skill(history)
    d = d[d["skill_id"].isin(s2i)].reset_index(drop=True)
    pos = positions(d, target_ids)
    rows = np.flatnonzero(pos >= 0)
    t, last = target_tensors(d, pos[rows], s2i, p2i)
    out = {}
    with torch.no_grad():
        for name in names:
            pr = np.full(len(rows), np.nan)
            for i in range(0, len(rows), batch_size):
                p = torch.sigmoid(models[name](_batch(t, slice(i, i + batch_size), device))).cpu().numpy()
                l = last[i:i + batch_size]
                ok = l >= 0
                pr[i:i + batch_size][ok] = p[np.flatnonzero(ok), l[ok]]
            out[name] = np.full(len(pos), np.nan)
            out[name][rows] = pr
    return out
