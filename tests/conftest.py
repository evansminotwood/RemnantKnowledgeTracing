import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_SKILLS, PER_SKILL = 4, 3   # problem p belongs to skill p // PER_SKILL


def _logs(rng, users, n_logs, first_id):
    """Students who get better at each skill with practice."""
    rows, pl = [], first_id
    for u in users:
        ability = rng.normal()
        seen = np.zeros(N_SKILLS)
        for _ in range(n_logs):
            p = int(rng.integers(N_SKILLS * PER_SKILL))
            k = p // PER_SKILL
            logit = ability + 0.4 * seen[k] - 1 + 0.3 * (p % PER_SKILL)
            rows.append((pl, u, k, 100 + p, int(rng.random() < 1 / (1 + np.exp(-logit)))))
            seen[k] += 1
            pl += int(rng.integers(1, 4))
    return pd.DataFrame(rows, columns=["id", "user_xid", "skill_id", "problem_id", "discrete_score"])


@pytest.fixture(scope="session")
def files(tmp_path_factory):
    """remnant.csv, rct_students.csv, rct_data.csv with the same column conventions as the real exports."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("data")
    remnant = _logs(rng, range(2000, 2300), 40, 10)
    remnant.to_csv(d / "remnant.csv", index=False)

    hist = _logs(rng, range(5000, 5120), 40, 10_000_000).rename(columns={"id": "pl_id"})
    hist["end_time"] = "2025-01-01 00:00:00"
    hist.drop(columns="skill_id").to_csv(d / "rct_students.csv", index=False)

    ev = hist.groupby("user_xid").apply(lambda g: g.iloc[[10, 20, 30]], include_groups=False).reset_index(0)
    rct = pd.DataFrame({"sl_id": np.arange(len(ev)) + 1, "pl_id": ev["pl_id"].to_numpy(),
                        "user_xid": ev["user_xid"].to_numpy(), "problem_id": ev["problem_id"].to_numpy(),
                        "class_id": ev["user_xid"].to_numpy() // 10,
                        "condition": rng.integers(0, 2, len(ev)),
                        "next_problem_across": rng.integers(0, 2, len(ev)).astype(float),
                        "prior_5_across": rng.random(len(ev)).round(1)})
    rct.to_csv(d / "rct_data.csv", index=False)
    return d
