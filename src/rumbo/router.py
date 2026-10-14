from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .classify import Classifier
from .providers import Msg, Provider
from .types import Decision, Reply, Route, RouteError, Score


@dataclass(frozen=True)
class Stage:
    clf: Classifier
    min_score: float = float("-inf")
    min_margin: float = 0.0


def top(ranked: Sequence[Score]) -> tuple[str, float, float]:
    if not ranked:
        return "", float("-inf"), 0.0
    nxt = ranked[1].val if len(ranked) > 1 else float("-inf")
    return ranked[0].route, ranked[0].val, ranked[0].val - nxt


class Router:
    def __init__(
        self, routes: Sequence[Route], stages: Sequence[Stage], fallback: str = ""
    ) -> None:
        if not stages:
            raise ValueError("router needs at least one stage")
        self.routes = {r.name: r for r in routes}
        if len(self.routes) != len(routes):
            raise ValueError("duplicate route names")
        self.stages = list(stages)
        self.fallback = fallback

    def route(self, text: str) -> Decision:
        t0 = time.perf_counter()
        ranked: Sequence[Score] = ()
        stg = self.stages[-1]
        for stg in self.stages:
            ranked = stg.clf.rank(text)
            name, score, margin = top(ranked)
            if name and score >= stg.min_score and margin >= stg.min_margin:
                return Decision(name, score, margin, stg.clf.name, _ms(t0), False, tuple(ranked))
        name, score, margin = top(ranked)
        return Decision(self.fallback, score, margin, stg.clf.name, _ms(t0), True, tuple(ranked))

    async def dispatch(
        self,
        text: str,
        providers: Mapping[str, Provider],
        history: Sequence[Msg] = (),
        timeout: float = 30.0,
    ) -> Reply:
        dec = self.route(text)
        rte = self.routes.get(dec.route)
        if rte is None:
            raise RouteError(f"no route for message (fallback={self.fallback!r})")
        chain = [n for n in (rte.target, *rte.backups) if n]
        if not chain:
            raise RouteError(f"route {rte.name!r} has no target")
        msgs = [*history, {"role": "user", "content": text}]
        errs: list[str] = []
        for i, name in enumerate(chain, 1):
            prv = providers.get(name)
            if prv is None:
                errs.append(f"{name}: not configured")
                continue
            try:
                out = await asyncio.wait_for(prv.chat(msgs), timeout)
            except Exception as exc:
                errs.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            return Reply(out, dec, name, i)
        raise RouteError("all providers failed: " + "; ".join(errs))


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
