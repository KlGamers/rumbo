from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .router import Router


@dataclass(frozen=True)
class Case:
    text: str
    want: str


@dataclass(frozen=True)
class PerRoute:
    n: int
    prec: float
    rec: float
    f1: float


@dataclass
class Report:
    n: int
    acc: float
    macro_f1: float
    p50_ms: float
    p95_ms: float
    routes: dict[str, PerRoute]
    stages: dict[str, float]
    confusion: dict[str, dict[str, int]] = field(repr=False, default_factory=dict)
    in_acc: float | None = None
    out_rec: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_md(self, worst: int = 5) -> str:
        rows = [
            "| metric | value |",
            "|---|---|",
            f"| cases | {self.n} |",
            f"| accuracy | {self.acc:.4f} |",
            f"| macro F1 | {self.macro_f1:.4f} |",
        ]
        if self.in_acc is not None:
            rows.append(f"| in-scope accuracy | {self.in_acc:.4f} |")
        if self.out_rec is not None:
            rows.append(f"| out-of-scope recall | {self.out_rec:.4f} |")
        rows += [
            f"| latency p50 | {self.p50_ms:.2f} ms |",
            f"| latency p95 | {self.p95_ms:.2f} ms |",
        ]
        for name, share in self.stages.items():
            rows.append(f"| decided by {name} | {share:.1%} |")
        out = [
            "\n".join(rows),
            "",
            "| route | n | precision | recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for name, pr in sorted(self.routes.items(), key=lambda kv: kv[1].f1)[:worst]:
            out.append(f"| {name} | {pr.n} | {pr.prec:.3f} | {pr.rec:.3f} | {pr.f1:.3f} |")
        return "\n".join(out)


def load(path: str | Path) -> list[Case]:
    out: list[Case] = []
    with open(path, encoding="utf-8") as fh:
        for num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                out.append(Case(obj["text"], obj["route"]))
            except KeyError as exc:
                raise ValueError(f"{path}:{num}: missing {exc}") from None
    return out


def score(
    want: Sequence[str],
    got: Sequence[str],
    ms: Sequence[float] = (),
    stages: Sequence[str] = (),
    out_lbl: str = "",
) -> Report:
    if len(want) != len(got) or not want:
        raise ValueError("want and got must be non-empty and the same length")
    conf: dict[str, Counter[str]] = defaultdict(Counter)
    for w, g in zip(want, got, strict=True):
        conf[w][g] += 1
    lbls = sorted(set(want) | set(got))
    per: dict[str, PerRoute] = {}
    for lb in lbls:
        tp = conf[lb][lb]
        fn = sum(conf[lb].values()) - tp
        fp = sum(conf[o][lb] for o in lbls if o != lb)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lb] = PerRoute(tp + fn, prec, rec, f1)
    seen = [pr for lb, pr in per.items() if lb in set(want)]
    hit = sum(conf[lb][lb] for lb in lbls)
    rep = Report(
        n=len(want),
        acc=hit / len(want),
        macro_f1=sum(pr.f1 for pr in seen) / len(seen),
        p50_ms=float(np.percentile(ms, 50)) if len(ms) else 0.0,
        p95_ms=float(np.percentile(ms, 95)) if len(ms) else 0.0,
        routes=per,
        stages={k: v / len(stages) for k, v in Counter(stages).most_common()} if stages else {},
        confusion={w: dict(c) for w, c in conf.items()},
    )
    if out_lbl and out_lbl in per:
        n_out = per[out_lbl].n
        n_in = rep.n - n_out
        rep.out_rec = per[out_lbl].rec if n_out else None
        rep.in_acc = (hit - conf[out_lbl][out_lbl]) / n_in if n_in else None
    return rep


def run(router: Router, cases: Iterable[Case]) -> Report:
    want, got, ms, stg = [], [], [], []
    for c in cases:
        dec = router.route(c.text)
        want.append(c.want)
        got.append(dec.route)
        ms.append(dec.ms)
        stg.append("fallback" if dec.fallback else dec.stage)
    return score(want, got, ms, stg, router.fallback)


def best_cut(
    vals: Sequence[float], ok: Sequence[bool], is_out: Sequence[bool]
) -> tuple[float, float]:
    v = np.asarray(vals, dtype=np.float64)
    good = np.asarray(ok, dtype=bool)
    out = np.asarray(is_out, dtype=bool)
    best = (float("-inf"), float(good.mean()))
    for cut in np.unique(v):
        keep = v >= cut
        acc = float(((keep & good) | (~keep & out)).mean())
        if acc > best[1]:
            best = (float(cut), acc)
    return best
