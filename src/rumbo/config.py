from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .classify import CentroidClassifier, Classifier, KnnClassifier, PatternClassifier
from .embed import Embedder, HashEmbedder, HttpEmbedder, StaticEmbedder
from .router import Router, Stage
from .types import Route


def _embedder(spec: dict[str, Any], routes: list[Route]) -> Embedder:
    kind = spec.get("embedder", "hash")
    if kind == "hash":
        emb = HashEmbedder(dim=int(spec.get("dim", 2048)))
        return emb.fit([e for r in routes for e in r.examples])
    if kind == "static":
        return StaticEmbedder(spec.get("model", "minishlab/potion-base-8M"))
    if kind == "http":
        return HttpEmbedder(spec["url"], spec["model"], os.environ.get(spec.get("key_env", ""), ""))
    raise ValueError(f"unknown embedder {kind!r}")


def _clf(spec: dict[str, Any], routes: list[Route]) -> Classifier:
    kind = spec.get("kind")
    if kind == "pattern":
        return PatternClassifier(routes)
    if kind == "centroid":
        return CentroidClassifier(routes, _embedder(spec, routes))
    if kind == "knn":
        return KnnClassifier(routes, _embedder(spec, routes), k=int(spec.get("k", 5)))
    raise ValueError(f"unknown stage kind {kind!r}")


def build(raw: dict[str, Any], clf: Callable[[dict, list[Route]], Classifier] = _clf) -> Router:
    routes = [
        Route(
            name=r["name"],
            desc=r.get("desc", ""),
            examples=tuple(r.get("examples", ())),
            patterns=tuple(r.get("patterns", ())),
            target=r.get("target", ""),
            backups=tuple(r.get("backups", ())),
        )
        for r in raw.get("routes", [])
    ]
    if not routes:
        raise ValueError("config has no routes")
    cfg = raw.get("router", {})
    specs = cfg.get("stages") or [{"kind": "knn"}]
    stages = [
        Stage(
            clf(s, routes),
            float(s.get("min_score", float("-inf"))),
            float(s.get("min_margin", 0.0)),
        )
        for s in specs
    ]
    return Router(routes, stages, cfg.get("fallback", ""))


def load(path: str | Path) -> Router:
    with open(path, "rb") as fh:
        return build(tomllib.load(fh))
