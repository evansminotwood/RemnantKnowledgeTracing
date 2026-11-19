import pandas as pd
import pytest

from analyzerct import columns

SPEC = "id,discrete_score,skill_id/problem_id,user_xid"


def test_parse_colspec():
    assert columns.parse_colspec(SPEC) == {"order": "id", "score": "discrete_score", "skill": "skill_id",
                                           "problem": "problem_id", "user": "user_xid"}
    with pytest.raises(ValueError):
        columns.parse_colspec("id,discrete_score,skill_id,user_xid")


def _infer(files, rct=None):
    remnant = pd.read_csv(files / "remnant.csv")
    students = pd.read_csv(files / "rct_students.csv")
    rct = pd.read_csv(files / "rct_data.csv") if rct is None else rct
    return columns.infer(columns.parse_colspec(SPEC), remnant, students, rct)


def test_infer_finds_every_column(files):
    s, notes = _infer(files)
    assert s["rct_students"] == {"order": "pl_id", "score": "discrete_score", "problem": "problem_id", "user": "user_xid"}
    assert s["rct_data"] == {"user": "user_xid", "log": "pl_id", "problem": "problem_id", "class": "class_id",
                             "treatment": "condition", "outcome": "next_problem_across", "prior5": "prior_5_across"}
    assert columns.missing(s) == []


def test_unsure_columns_are_left_blank(files, tmp_path):
    rct = pd.read_csv(files / "rct_data.csv").rename(columns={"condition": "strategy"})
    rct["strategy"] = rct["strategy"].map({0: "[]", 1: "[TS1]"})
    s, notes = _infer(files, rct)
    assert s["rct_data"]["treatment"] is None
    assert columns.missing(s) == ["rct_data.treatment"]
    path = tmp_path / "schema.yaml"
    columns.write(s, path, notes)
    assert "UNSURE" in path.read_text()


def test_schema_round_trip(files, tmp_path):
    s, notes = _infer(files)
    s["rct_data"]["treatment"] = "strategy != '[]'"
    s["exclude"] = ["user_xid <= 1000"]
    path = tmp_path / "schema.yaml"
    columns.write(s, path, notes)
    back = columns.read(path)
    assert back["rct_data"] == s["rct_data"] and back["rct_students"] == s["rct_students"]
    assert back["exclude"] == ["user_xid <= 1000"]
