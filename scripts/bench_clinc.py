from __future__ import annotations

import argparse
import json
import platform
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rumbo import (
    Case,
    CentroidClassifier,
    ConcatEmbedder,
    HashEmbedder,
    KnnClassifier,
    LinearClassifier,
    Route,
    Router,
    Stage,
    best_cut,
    run,
)
from rumbo.router import top

SRC = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
OUT = "oos"


class Memo:
    def __init__(self, emb) -> None:
        self.emb = emb
        self.memo: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        miss = [t for t in dict.fromkeys(texts) if t not in self.memo]
        if miss:
            for t, v in zip(miss, self.emb.embed(miss), strict=True):
                self.memo[t] = v
        return np.stack([self.memo[t] for t in texts])


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


def tune(clf, val: list[Case]) -> tuple[float, float]:
    got, sc = [], []
    for ranked in clf.rank_many([c.text for c in val]):
        name, score, _ = top(ranked)
        got.append(name)
        sc.append(score)
    ok = [g == c.want for g, c in zip(got, val, strict=True)]
    return best_cut(sc, ok, [c.want == OUT for c in val])


def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("--data", default="data/clinc")
    par.add_argument("--out", default="bench/RESULTS.md")
    par.add_argument("--static", default="minishlab/potion-base-8M")
    par.add_argument("--onnx", default="BAAI/bge-small-en-v1.5")
    par.add_argument("--l2", default="1e-4,1e-5,1e-6")
    args = par.parse_args()

    data, doms = fetch(Path(args.data))
    dom = {intent: d for d, intents in doms.items() for intent in intents}
    by: dict[str, list[str]] = {d: [] for d in doms}
    for text, lb in data["train"]:
        by[dom[lb]].append(text)
    routes = [Route(d, examples=tuple(ex)) for d, ex in by.items()]
    seen = routes + [Route(OUT, examples=tuple(t for t, _ in data["oos_train"]))]
    val = cases(data, dom, "val", "oos_val")
    test = cases(data, dom, "test", "oos_test")

    hsh = HashEmbedder(dim=8192).fit([t for ex in by.values() for t in ex])
    embs = {"hash": Memo(hsh)}
    try:
        from rumbo import StaticEmbedder

        embs["static"] = Memo(StaticEmbedder(args.static))
    except ImportError:
        pass
    try:
        from rumbo import OnnxEmbedder

        embs["onnx + hash"] = Memo(ConcatEmbedder(OnnxEmbedder(args.onnx), hsh))
    except ImportError:
        pass

    sets: dict[str, list[Stage]] = {}
    cen = CentroidClassifier(routes, embs["hash"])
    sets["centroid / hash"] = [Stage(cen, tune(cen, val)[0])]
    knn = KnnClassifier(seen, embs["hash"], k=5)
    sets["kNN / hash"] = [Stage(knn, tune(knn, val)[0])]
    if "static" in embs:
        kst = KnnClassifier(seen, embs["static"], k=5)
        sets["kNN / static"] = [Stage(kst, tune(kst, val)[0])]
    picks = []
    for tag in [t for t in ("hash", "onnx + hash") if t in embs]:
        best = None
        for l2 in (float(v) for v in args.l2.split(",")):
            lin = LinearClassifier(seen, embs[tag], l2=l2)
            cut, acc = tune(lin, val)
            if best is None or acc > best[0]:
                best = (acc, l2, lin, cut)
        acc, l2, lin, cut = best
        picks.append(f"linear / {tag}: l2={l2:g}, cut={cut:.3f}, validation accuracy {acc:.4f}")
        sets[f"linear / {tag}"] = [Stage(lin, cut)]

    rows = [
        "| router | accuracy | in-scope | out-of-scope recall | macro F1 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, stages in sets.items():
        for st in stages:
            emb = getattr(st.clf, "emb", None)
            if isinstance(emb, Memo):
                st.clf.emb = emb.emb
        rep = run(Router(seen, stages, fallback=OUT), test)
        rows.append(
            f"| {name} | {rep.acc:.4f} | {rep.in_acc:.4f} | {rep.out_rec:.4f} | {rep.macro_f1:.4f} "
            f"| {rep.p50_ms:.2f} | {rep.p95_ms:.2f} |"
        )
        print(rows[-1], flush=True)

    head = [
        "# CLINC150 domain routing",
        "",
        f"- routes: {len(routes)} domains, {sum(len(r.examples) for r in routes)} examples",
        f"- thresholds and l2 tuned on: {len(val)} validation cases",
        f"- reported on: {len(test)} test cases, {sum(c.want == OUT for c in test)} out of scope",
        f"- machine: {platform.machine()}, {platform.system()}, Python {platform.python_version()}",
        *[f"- {p}" for p in picks],
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(head + rows) + "\n")
    print("\n".join(head))


if __name__ == "__main__":
    main()
