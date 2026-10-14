from __future__ import annotations

import argparse
import json
import platform
import urllib.request
from pathlib import Path

import numpy as np

from rumbo import (
    Case,
    CentroidClassifier,
    HashEmbedder,
    KnnClassifier,
    Route,
    Router,
    Stage,
    best_cut,
    run,
)
from rumbo.router import top

SRC = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
OUT = "oos"


def fetch(root: Path) -> tuple[dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    objs = []
    for name in ("data_full.json", "domains.json"):
        dst = root / name
        if not dst.exists():
            urllib.request.urlretrieve(SRC + name, dst)
        objs.append(json.loads(dst.read_text()))
    return objs[0], objs[1]


def cases(data: dict, dom: dict[str, str], *parts: str) -> list[Case]:
    return [Case(t, dom.get(lb, OUT)) for p in parts for t, lb in data[p]]


def ranks(clf, cs: list[Case]) -> tuple[list[str], np.ndarray, np.ndarray]:
    got, val, mrg = [], [], []
    for ranked in clf.rank_many([c.text for c in cs]):
        name, score, margin = top(ranked)
        got.append(name)
        val.append(score)
        mrg.append(margin)
    return got, np.asarray(val), np.asarray(mrg)


def tune(clf, val: list[Case]) -> float:
    got, sc, _ = ranks(clf, val)
    ok = [g == c.want for g, c in zip(got, val, strict=True)]
    return best_cut(sc, ok, [c.want == OUT for c in val])[0]


def tune_margin(a, cut_a: float, b, cut_b: float, val: list[Case], slack: float) -> float:
    ga, sa, ma = ranks(a, val)
    gb, sb, _ = ranks(b, val)
    want = np.asarray([c.want for c in val])
    pick_b = np.where(sb >= cut_b, np.asarray(gb), OUT)
    goal = float((pick_b == want).mean()) - slack
    for m in np.quantile(ma, np.linspace(0, 1, 101)):
        use_a = (sa >= cut_a) & (ma >= m)
        acc = float((np.where(use_a, np.asarray(ga), pick_b) == want).mean())
        if acc >= goal:
            return float(m)
    return float("inf")


def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("--data", default="data/clinc")
    par.add_argument("--out", default="bench/RESULTS.md")
    par.add_argument("--static", default="minishlab/potion-base-8M")
    par.add_argument("--slack", type=float, default=0.005)
    args = par.parse_args()

    data, doms = fetch(Path(args.data))
    dom = {intent: d for d, intents in doms.items() for intent in intents}
    by: dict[str, list[str]] = {d: [] for d in doms}
    for text, lb in data["train"]:
        by[dom[lb]].append(text)
    routes = [Route(d, examples=tuple(ex)) for d, ex in by.items()]
    val = cases(data, dom, "val", "oos_val")
    test = cases(data, dom, "test", "oos_test")

    seen = [Route(OUT, examples=tuple(t for t, _ in data["oos_train"]))]
    hsh = HashEmbedder(dim=2048).fit([t for ex in by.values() for t in ex])
    embs: dict[str, object] = {"hash": hsh}
    try:
        from rumbo import StaticEmbedder

        embs["static"] = StaticEmbedder(args.static)
    except ImportError:
        pass

    sets: dict[str, list[Stage]] = {}
    for tag, emb in embs.items():
        cen = CentroidClassifier(routes, emb)
        sets[f"centroid / {tag}"] = [Stage(cen, tune(cen, val))]
        knn = KnnClassifier(routes, emb, k=5)
        sets[f"kNN / {tag}"] = [Stage(knn, tune(knn, val))]
        full = KnnClassifier(routes + seen, emb, k=5)
        cut = tune(full, val)
        sets[f"kNN / {tag} + out-of-scope examples"] = [Stage(full, cut)]
        cen2 = CentroidClassifier(routes + seen, emb)
        cut2 = tune(cen2, val)
        mrg = tune_margin(cen2, cut2, full, cut, val, args.slack)
        sets[f"cascade / {tag}: centroid then kNN"] = [Stage(cen2, cut2, mrg), Stage(full, cut)]

    rows = [
        "| router | accuracy | in-scope | out-of-scope recall | macro F1 | p50 ms | p95 ms "
        "| escalated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, stages in sets.items():
        rep = run(Router(routes + seen, stages, fallback=OUT), test)
        esc = "-"
        if len(stages) > 1:
            esc = f"{1 - rep.stages.get(stages[0].clf.name, 0.0):.1%}"
        rows.append(
            f"| {name} | {rep.acc:.4f} | {rep.in_acc:.4f} | {rep.out_rec:.4f} | {rep.macro_f1:.4f} "
            f"| {rep.p50_ms:.2f} | {rep.p95_ms:.2f} | {esc} |"
        )
        print(rows[-1], flush=True)

    head = [
        "# CLINC150 domain routing",
        "",
        f"- routes: {len(routes)} domains, {sum(len(r.examples) for r in routes)} examples",
        f"- thresholds tuned on: {len(val)} validation cases",
        f"- reported on: {len(test)} test cases, {sum(c.want == OUT for c in test)} out of scope",
        f"- machine: {platform.machine()}, {platform.system()}, Python {platform.python_version()}",
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(head + rows) + "\n")


if __name__ == "__main__":
    main()
