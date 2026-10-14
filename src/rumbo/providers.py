from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

import httpx

Msg = dict[str, str]


class Provider(Protocol):
    async def chat(self, msgs: Sequence[Msg]) -> str: ...


class FnProvider:
    def __init__(self, fn: Callable[[Sequence[Msg]], Awaitable[str]]) -> None:
        self.fn = fn

    async def chat(self, msgs: Sequence[Msg]) -> str:
        return await self.fn(msgs)


class OpenAICompat:
    def __init__(
        self,
        url: str,
        model: str,
        key: str = "",
        timeout: float = 60.0,
        retries: int = 2,
        wait: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        hdr = {"Authorization": f"Bearer {key}"} if key else {}
        self.cli = httpx.AsyncClient(
            base_url=url.rstrip("/"), headers=hdr, timeout=timeout, transport=transport
        )
        self.model = model
        self.retries = retries
        self.wait = wait

    async def chat(self, msgs: Sequence[Msg]) -> str:
        body = {"model": self.model, "messages": list(msgs)}
        err: Exception | None = None
        for i in range(self.retries + 1):
            if i:
                await asyncio.sleep(self.wait * 2 ** (i - 1))
            try:
                res = await self.cli.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                err = exc
                continue
            if res.status_code == 429 or res.status_code >= 500:
                err = httpx.HTTPStatusError(
                    f"status {res.status_code}", request=res.request, response=res
                )
                continue
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        assert err is not None
        raise err

    async def aclose(self) -> None:
        await self.cli.aclose()
