import sys
import types

import numpy as np
import pytest

from rumbo import ConcatEmbedder, HashEmbedder, LinearClassifier, OnnxEmbedder, Route
from rumbo import config as cfg
from rumbo.classify import fit, softmax

ROUTES = [
    Route("pay", examples=("pay my bill", "send money to a friend", "transfer funds", "pay rent")),
    Route(
        "cook",
        examples=(
            "how do i bake bread",
            "recipe for pasta",
            "what can i cook tonight",
            "boil eggs",
        ),
    ),
    Route(
        "trip",
        examples=(
            "book a flight",
            "find a hotel in rome",
            "rent a car at the airport",
            "train tickets",
        ),
    ),
]


def test_softmax_is_stable():
    p = softmax(np.array([[1000.0, 1000.0], [0.0, -1000.0]]))
    assert np.allclose(p.sum(axis=1), 1) and np.allclose(p[0], 0.5) and p[1, 0] > 0.999


def test_fit_learns_separable_data():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2, 0.3, (40, 3)), rng.normal(2, 0.3, (40, 3))]).astype(np.float32)
    y = np.array([0] * 40 + [1] * 40)
    w, b = fit(x, y, 2, 1e-4, 200, 0.1)
    assert ((x @ w + b).argmax(axis=1) == y).mean() == 1.0


def test_linear_classifier():
    clf = LinearClassifier(
        ROUTES, HashEmbedder(dim=1024).fit([e for r in ROUTES for e in r.examples])
    )
    assert clf.rank("i want to transfer money")[0].route == "pay"
    assert clf.rank("pasta recipe please")[0].route == "cook"
    assert clf.rank("book me a hotel")[0].route == "trip"
    probs = clf.probs(["pay my bill", "bake bread", "flight"], batch=2)
    assert probs.shape == (3, 3) and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    many = clf.rank_many(["pay my bill", "bake bread"])
    assert [r[0].route for r in many] == ["pay", "cook"]
    assert clf.probs([]).shape == (0, 3)


def test_linear_needs_two_routes():
    with pytest.raises(ValueError):
        LinearClassifier([ROUTES[0], Route("empty")], HashEmbedder(dim=64))


def test_concat():
    a, b = HashEmbedder(dim=64), HashEmbedder(dim=32)
    v = ConcatEmbedder(a, b).embed(["hello there"])
    assert v.shape == (1, 96) and np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        ConcatEmbedder()


def test_onnx_embedder_with_fake_backend(monkeypatch):
    class Fake:
        def __init__(self, name):
            self.name = name

        def embed(self, texts, batch_size):
            for t in texts:
                yield [float(len(t)), 1.0, 0.0]

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=Fake))
    emb = OnnxEmbedder("tiny")
    out = emb.embed(["ab", "abcd"])
    assert out.shape == (2, 3) and np.allclose(np.linalg.norm(out, axis=1), 1)
    assert emb.embed([]).shape == (0, 0)


def test_config_linear_and_concat(monkeypatch):
    raw = {
        "router": {
            "fallback": "trip",
            "stages": [{"kind": "linear", "embedder": ["hash", "hash"], "dim": 256, "epochs": 50}],
        },
        "routes": [{"name": r.name, "examples": list(r.examples)} for r in ROUTES],
    }
    rtr = cfg.build(raw)
    clf = rtr.stages[0].clf
    assert isinstance(clf, LinearClassifier) and isinstance(clf.emb, ConcatEmbedder)
    assert rtr.route("send money to my friend").route == "pay"
