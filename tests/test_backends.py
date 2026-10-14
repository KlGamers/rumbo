import sys
import types

import httpx
import numpy as np
import pytest

from rumbo import (
    CentroidClassifier,
    HttpEmbedder,
    KnnClassifier,
    LinearClassifier,
    OnnxEmbedder,
    OpenAICompat,
    StaticEmbedder,
)
from rumbo import config as cfg

ROUTES = [
    {"name": "pay", "examples": ["pay my bill", "transfer funds"]},
    {"name": "cook", "examples": ["bake bread", "pasta recipe"]},
]


class FakeStatic:
    def __init__(self, *_):
        pass

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def encode(self, texts):
        return [[1.0, float(len(t))] for t in texts]


class FakeOnnx:
    def __init__(self, name):
        pass

    def embed(self, texts, batch_size):
        for t in texts:
            yield [float("pay" in t), float("bread" in t or "pasta" in t), 1.0]


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setitem(sys.modules, "model2vec", types.SimpleNamespace(StaticModel=FakeStatic))
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeOnnx))


def test_static_embedder(fakes):
    v = StaticEmbedder("m").embed(["ab", "abcd"])
    assert v.shape == (2, 2) and np.allclose(np.linalg.norm(v, axis=1), 1)


def test_http_embedder_orders_rows():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]},
        )

    emb = HttpEmbedder("https://e.test/v1/", "m", key="k", transport=httpx.MockTransport(handler))
    v = emb.embed(["a", "b"])
    assert np.allclose(v, [[1, 0], [0, 1]]) and seen["auth"] == "Bearer k"
    bad = HttpEmbedder(
        "https://e.test/v1", "m", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    with pytest.raises(httpx.HTTPStatusError):
        bad.embed(["a"])


def test_config_embedders(fakes, monkeypatch):
    def build(stage):
        return cfg.build({"routes": ROUTES, "router": {"stages": [stage]}}).stages[0].clf

    assert isinstance(build({"kind": "centroid"}), CentroidClassifier)
    assert isinstance(build({"kind": "knn", "embedder": "static"}).emb, StaticEmbedder)
    assert isinstance(build({"kind": "linear", "embedder": "onnx", "epochs": 20}).emb, OnnxEmbedder)
    monkeypatch.setenv("MY_KEY", "secret")
    captured = {}

    def init(self, url, model, key="", **kw):
        captured.update(url=url, model=model, key=key)

    monkeypatch.setattr(HttpEmbedder, "__init__", init)
    monkeypatch.setattr(
        HttpEmbedder, "embed", lambda self, texts: np.eye(len(texts), 4, dtype=np.float32)
    )
    clf = build(
        {
            "kind": "knn",
            "embedder": "http",
            "url": "https://e.test",
            "model": "m",
            "key_env": "MY_KEY",
        }
    )
    assert isinstance(clf, KnnClassifier) and captured == {
        "url": "https://e.test",
        "model": "m",
        "key": "secret",
    }


def test_linear_with_fake_onnx(fakes):
    from rumbo import Route

    rts = [Route(r["name"], examples=tuple(r["examples"])) for r in ROUTES]
    clf = LinearClassifier(rts, OnnxEmbedder("x"), epochs=100)
    assert clf.rank("pay rent")[0].route == "pay" and clf.rank("bread")[0].route == "cook"


async def test_openai_compat_retries_transport_errors():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    prv = OpenAICompat("https://l.test/v1", "m", wait=0.001, transport=httpx.MockTransport(handler))
    assert await prv.chat([]) == "ok" and len(calls) == 2
    dead = OpenAICompat(
        "https://l.test/v1",
        "m",
        retries=1,
        wait=0.001,
        transport=httpx.MockTransport(
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("x", request=r))
        ),
    )
    with pytest.raises(httpx.ConnectError):
        await dead.chat([])
