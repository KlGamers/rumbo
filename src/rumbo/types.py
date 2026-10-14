from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Route:
    name: str
    desc: str = ""
    examples: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    target: str = ""
    backups: tuple[str, ...] = ()


@dataclass(frozen=True)
class Score:
    route: str
    val: float


@dataclass(frozen=True)
class Decision:
    route: str
    score: float
    margin: float
    stage: str
    ms: float
    fallback: bool = False
    ranked: tuple[Score, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class Reply:
    text: str
    decision: Decision
    provider: str
    tries: int


class RouteError(Exception):
    pass
