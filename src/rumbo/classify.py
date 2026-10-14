from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from .embed import Embedder, unit
from .types import Route, Score


class Classifier(Protocol):
    name: str

    def rank(self, text: str) -> list[Score]: ...


def order(scores: dict[str, float]) -> list[Score]:
    return [Score(r, float(v)) for r, v in sorted(scores.items(), key=lambda kv: -kv[1])]


class PatternClassifier:
    name = "pattern"

    def __init__(self, routes: Sequence[Route]) -> None:
        self.rx = {r.name: [re.compile(p, re.I) for p in r.patterns] for r in routes if r.patterns}

    def rank(self, text: str) -> list[Score]:
        hits = {name: sum(1 for p in pats if p.search(text)) for name, pats in self.rx.items()}
        return order({name: float(n) for name, n in hits.items() if n})


class _VecBase:
    def __init__(self, routes: Sequence[Route], emb: Embedder) -> None:
        self.emb = emb
        self.names: list[str] = []
        self.spans: list[tuple[int, int]] = []
        texts: list[str] = []
        for r in routes:
            if not r.examples:
                continue
            self.names.append(r.name)
            self.spans.append((len(texts), len(texts) + len(r.examples)))
            texts.extend(r.examples)
        if not texts:
            raise ValueError("no route has examples")
        self.mat = emb.embed(texts)

    def rank(self, text: str) -> list[Score]:
        return self.rank_many([text])[0]

    def rank_many(self, texts: Sequence[str], batch: int = 256) -> list[list[Score]]:
        out: list[list[Score]] = []
        for i in range(0, len(texts), batch):
            qry = self.emb.embed(texts[i : i + batch])
            for row in self._score(qry):
                out.append(order(dict(zip(self.names, row, strict=True))))
        return out

    def _score(self, qry: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CentroidClassifier(_VecBase):
    name = "centroid"

    def __init__(self, routes: Sequence[Route], emb: Embedder) -> None:
        super().__init__(routes, emb)
        self.cen = unit(np.stack([self.mat[a:b].mean(axis=0) for a, b in self.spans]))

    def _score(self, qry: np.ndarray) -> np.ndarray:
        return qry @ self.cen.T


class KnnClassifier(_VecBase):
    name = "knn"

    def __init__(self, routes: Sequence[Route], emb: Embedder, k: int = 5) -> None:
        super().__init__(routes, emb)
        self.k = k

    def _score(self, qry: np.ndarray) -> np.ndarray:
        sims = qry @ self.mat.T
        out = np.empty((qry.shape[0], len(self.spans)), dtype=np.float32)
        for col, (a, b) in enumerate(self.spans):
            part = sims[:, a:b]
            k = min(self.k, b - a)
            top = np.partition(part, part.shape[1] - k, axis=1)[:, -k:]
            out[:, col] = top.mean(axis=1)
        return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LinearClassifier:
    name = "linear"

    def __init__(
        self,
        routes: Sequence[Route],
        emb: Embedder,
        l2: float = 1e-5,
        epochs: int = 300,
        lr: float = 0.05,
    ) -> None:
        self.emb = emb
        self.names = [r.name for r in routes if r.examples]
        if len(self.names) < 2:
            raise ValueError("linear classifier needs at least two routes with examples")
        texts = [e for r in routes if r.examples for e in r.examples]
        lbl = np.concatenate(
            [np.full(len(r.examples), i) for i, r in enumerate(r for r in routes if r.examples)]
        )
        self.w, self.b = fit(emb.embed(texts), lbl, len(self.names), l2, epochs, lr)

    def probs(self, texts: Sequence[str], batch: int = 256) -> np.ndarray:
        out = [
            softmax(self.emb.embed(texts[i : i + batch]) @ self.w + self.b)
            for i in range(0, len(texts), batch)
        ]
        return np.vstack(out) if out else np.zeros((0, len(self.names)), dtype=np.float32)

    def rank(self, text: str) -> list[Score]:
        return self.rank_many([text])[0]

    def rank_many(self, texts: Sequence[str], batch: int = 256) -> list[list[Score]]:
        return [order(dict(zip(self.names, row, strict=True))) for row in self.probs(texts, batch)]


def fit(
    x: np.ndarray, y: np.ndarray, k: int, l2: float, epochs: int, lr: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    w = np.zeros((dim, k), dtype=np.float32)
    b = np.zeros(k, dtype=np.float32)
    hot = np.eye(k, dtype=np.float32)[y]
    state = [np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b)]
    for t in range(1, epochs + 1):
        g = (softmax(x @ w + b) - hot) / n
        gw = x.T @ g + l2 * w
        gb = g.sum(axis=0)
        for prm, grd, m, v in ((w, gw, state[0], state[1]), (b, gb, state[2], state[3])):
            m *= 0.9
            m += 0.1 * grd
            v *= 0.999
            v += 0.001 * grd * grd
            prm -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return w, b


PROMPT = (
    "You are an intent router. Pick the single best route for the user message.\n"
    'Reply with JSON only: {{"route": "<name>", "confidence": <0..1>}}.\n'
    'Use "none" when no route applies.\n\nRoutes:\n{routes}'
)


class LlmClassifier:
    name = "llm"

    def __init__(self, routes: Sequence[Route], ask: Callable[[str, str], str]) -> None:
        self.names = {r.name for r in routes}
        self.sys = PROMPT.format(routes="\n".join(f"- {r.name}: {r.desc}" for r in routes))
        self.ask = ask

    def rank(self, text: str) -> list[Score]:
        raw = self.ask(self.sys, text)
        hit = re.search(r"\{.*\}", raw, re.S)
        if not hit:
            return []
        try:
            obj = json.loads(hit.group(0))
        except json.JSONDecodeError:
            return []
        name = obj.get("route")
        if name not in self.names:
            return []
        try:
            conf = float(obj.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        return [Score(name, min(max(conf, 0.0), 1.0))]
