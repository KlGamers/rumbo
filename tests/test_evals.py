import json

import pytest

from rumbo import Case, best_cut, evals, score
from rumbo import config as cfg
from rumbo.cli import main


def test_score_metrics():
    want = ["a", "a", "a", "b", "b", "oos"]
    got = ["a", "a", "b", "b", "oos", "oos"]
    rep = score(want, got, ms=[1, 2, 3, 4, 5, 6], stages=["knn"] * 5 + ["fallback"], out_lbl="oos")
    assert rep.n == 6 and rep.acc == pytest.approx(4 / 6)
    assert rep.routes["a"].prec == 1.0 and rep.routes["a"].rec == pytest.approx(2 / 3)
    assert rep.routes["b"].prec == 0.5 and rep.routes["b"].rec == 0.5
    assert rep.routes["oos"].prec == 0.5 and rep.out_rec == 1.0
    assert rep.in_acc == pytest.approx(3 / 5)
    f1 = [0.8, 0.5, 2 / 3]
    assert rep.macro_f1 == pytest.approx(sum(f1) / 3)
    assert rep.confusion["a"] == {"a": 2, "b": 1}
    assert rep.p50_ms == 3.5 and rep.stages["knn"] == pytest.approx(5 / 6)
    assert "| accuracy | 0.6667 |" in rep.to_md()
    assert json.dumps(rep.to_dict())


def test_score_rejects_bad_input():
    with pytest.raises(ValueError):
        score([], [])
    with pytest.raises(ValueError):
        score(["a"], ["a", "b"])


def test_macro_f1_ignores_labels_never_expected():
    rep = score(["a", "a"], ["a", "zzz"])
    assert rep.macro_f1 == pytest.approx(2 / 3)


def test_best_cut():
    vals = [0.9, 0.8, 0.3, 0.2, 0.1]
    ok = [True, True, False, False, False]
    out = [False, False, True, True, False]
    cut, acc = best_cut(vals, ok, out)
    assert cut == 0.8 and acc == pytest.approx(4 / 5)
    assert best_cut([0.5, 0.6], [True, True], [False, False]) == (float("-inf"), 1.0)


def test_load_reports_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "a", "route": "x"}\n\n{"text": "b"}\n')
    with pytest.raises(ValueError, match="d.jsonl:3"):
        evals.load(p)


def test_config_and_cli(tmp_path, capsys):
    rtr = cfg.load("examples/routes.toml")
    assert rtr.fallback == "general" and [s.clf.name for s in rtr.stages] == ["pattern", "knn"]
    assert rtr.route("i got this traceback").stage == "pattern"
    rep = evals.run(rtr, evals.load("examples/sample.jsonl"))
    assert rep.acc >= 0.9

    out = tmp_path / "rep.json"
    code = main(
        [
            "eval",
            "examples/sample.jsonl",
            "-c",
            "examples/routes.toml",
            "--json",
            str(out),
            "--min-acc",
            "0.9",
        ]
    )
    assert code == 0 and json.loads(out.read_text())["n"] == 11
    assert (
        main(["eval", "examples/sample.jsonl", "-c", "examples/routes.toml", "--min-acc", "1.01"])
        == 1
    )
    capsys.readouterr()
    assert main(["route", "solve 4x = 12", "-c", "examples/routes.toml"]) == 0
    assert json.loads(capsys.readouterr().out)["route"] == "math"


def test_config_errors():
    with pytest.raises(ValueError, match="no routes"):
        cfg.build({})
    with pytest.raises(ValueError, match="unknown stage"):
        cfg.build(
            {"routes": [{"name": "a", "examples": ["x"]}], "router": {"stages": [{"kind": "zzz"}]}}
        )
    with pytest.raises(ValueError, match="unknown embedder"):
        cfg.build(
            {
                "routes": [{"name": "a", "examples": ["x"]}],
                "router": {"stages": [{"kind": "knn", "embedder": "zzz"}]},
            }
        )


def test_case_is_hashable():
    assert len({Case("a", "x"), Case("a", "x")}) == 1
