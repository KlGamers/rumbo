from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np

_WORD = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def unit(mat: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(mat, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return mat / nrm


class HashEmbedder:
    def __init__(self, dim: int = 2048, lo: int = 3, hi: int = 5) -> None:
        self.dim = dim
        self.lo = lo
        self.hi = hi
        self.idf: np.ndarray | None = None

    def feats(self, text: str) -> Counter[int]:
        words = _WORD.findall(text.lower())
        cnt: Counter[int] = Counter()
        for w in words:
            cnt[self._slot("w:" + w)] += 1
            pad = f"<{w}>"
            for n in range(self.lo, self.hi + 1):
                for i in range(len(pad) - n + 1):
                    cnt[self._slot(pad[i : i + n])] += 1
        for a, b in zip(words, words[1:], strict=False):
            cnt[self._slot(f"b:{a} {b}")] += 1
        return cnt

    def _slot(self, tok: str) -> int:
        return zlib.crc32(tok.encode()) % self.dim

    def _raw(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            cnt = self.feats(text)
            if cnt:
                idx = np.fromiter(cnt.keys(), dtype=np.int64, count=len(cnt))
                val = np.fromiter(cnt.values(), dtype=np.float32, count=len(cnt))
                out[row, idx] = 1.0 + np.log(val)
        return out

    def fit(self, texts: Sequence[str]) -> HashEmbedder:
        raw = self._raw(texts)
        df = (raw > 0).sum(axis=0)
        self.idf = (np.log((1 + len(texts)) / (1 + df)) + 1.0).astype(np.float32)
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raw = self._raw(texts)
        if self.idf is not None:
            raw *= self.idf
        return unit(raw)


class ConcatEmbedder:
    def __init__(self, *parts: Embedder) -> None:
        if not parts:
            raise ValueError("need at least one embedder")
        self.parts = parts

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.hstack([p.embed(texts) for p in self.parts]) / np.sqrt(len(self.parts))


class OnnxEmbedder:
    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", batch: int = 256) -> None:
        from fastembed import TextEmbedding

        self.mdl = TextEmbedding(name)
        self.batch = batch

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self.mdl.embed(list(texts), batch_size=self.batch)
        return unit(np.asarray(list(vecs), dtype=np.float32))


class StaticEmbedder:
    def __init__(self, name: str = "minishlab/potion-base-8M") -> None:
        from model2vec import StaticModel

        self.mdl = StaticModel.from_pretrained(name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return unit(np.asarray(self.mdl.encode(list(texts)), dtype=np.float32))


class HttpEmbedder:
    def __init__(
        self,
        url: str,
        model: str,
        key: str = "",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        hdr = {"Authorization": f"Bearer {key}"} if key else {}
        self.cli = httpx.Client(
            base_url=url.rstrip("/"), headers=hdr, timeout=timeout, transport=transport
        )
        self.model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        res = self.cli.post("/embeddings", json={"model": self.model, "input": list(texts)})
        res.raise_for_status()
        rows = sorted(res.json()["data"], key=lambda r: r["index"])
        return unit(np.asarray([r["embedding"] for r in rows], dtype=np.float32))
