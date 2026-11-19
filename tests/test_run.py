import pandas as pd
import pytest

import analyzeRCT
from analyzerct.models import dkt

SPEC = "id,discrete_score,skill_id/problem_id,user_xid"


def _run(files, out, *flags, schema=None):
    argv = [str(files / "remnant.csv"), SPEC, str(files / "rct_students.csv"), str(files / "rct_data.csv"),
            *flags, "--out", str(out), "--schema", str(schema or out.parent / f"{out.name}_schema.yaml")]
    analyzeRCT.main(argv)


def test_stops_until_unsure_columns_are_filled_in(files, tmp_path):
    rct = pd.read_csv(files / "rct_data.csv").rename(columns={"condition": "strategy"})
    rct["strategy"] = rct["strategy"].map({0: "[]", 1: "[TS1]"})
    rct.to_csv(tmp_path / "rct_data.csv", index=False)
    for f in ["remnant.csv", "rct_students.csv"]:
        (tmp_path / f).write_bytes((files / f).read_bytes())
    schema = tmp_path / "schema.yaml"
    with pytest.raises(SystemExit, match="rct_data.treatment"):
        _run(tmp_path, tmp_path / "out", "--Prior5", schema=schema)
    assert schema.exists() and not (tmp_path / "out").exists()

    schema.write_text(schema.read_text().replace("  treatment: ", "  treatment: strategy != '[]'", 1))
    _run(tmp_path, tmp_path / "out", "--Prior5", schema=schema)
    pred = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert pred["treatment"].mean() == pytest.approx(rct["strategy"].ne("[]").mean())


def test_every_model_end_to_end(files, tmp_path, monkeypatch):
    monkeypatch.setattr(dkt, "EPOCHS", 3)
    out = tmp_path / "out"
    flags = ["--Prior5", "--BKT", "--DKT", "--DKT-P", "--IRT"]
    _run(files, out, *flags)
    for f in ["schema.yaml", "predictions.csv", "effects.csv", "summary.md", "models/bkt_params.csv",
              "models/dkt_models.pt", "models/irt_params.csv"]:
        assert (out / f).exists(), f
    pred = pd.read_csv(out / "predictions.csv")
    for c in ["prior5", "bkt", "dkt", "dkt_p", "irt"]:
        assert pred[c].notna().mean() > 0.9 and pred[c].dropna().between(0, 1).all(), c
    eff = pd.read_csv(out / "effects.csv")
    assert set(eff["control"]) == {"Unadjusted", "Prior5", "BKT", "DKT", "DKT-P", "IRT"}
    assert "| DKT-P |" in (out / "summary.md").read_text()

    # a second run into the same folder reuses the trained models
    before = (out / "models" / "dkt_models.pt").stat().st_mtime
    _run(files, out, "--DKT", "--DKT-P")
    assert (out / "models" / "dkt_models.pt").stat().st_mtime == before
