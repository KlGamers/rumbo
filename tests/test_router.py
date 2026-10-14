import asyncio

import httpx
import pytest

from rumbo import FnProvider, OpenAICompat, Route, RouteError, Router, Score, Stage


class Fixed:
    def __init__(self, name, table):
        self.name = name
        self.table = table
        self.calls = 0

    def rank(self, text):
        self.calls += 1
        return [Score(r, v) for r, v in self.table.get(text, [])]


ROUTES = [Route("a", target="p1", backups=("p2",)), Route("b", target="p2"), Route("c")]


def test_single_stage_and_fallback():
    clf = Fixed("s1", {"x": [("a", 0.9), ("b", 0.2)], "y": [("a", 0.3), ("b", 0.2)]})
    rtr = Router(ROUTES, [Stage(clf, min_score=0.5)], fallback="b")
    dec = rtr.route("x")
    assert (dec.route, dec.stage, dec.fallback) == ("a", "s1", False)
    assert dec.margin == pytest.approx(0.7) and dec.ms >= 0
    low = rtr.route("y")
    assert (low.route, low.fallback) == ("b", True)
    assert rtr.route("unknown").fallback


def test_cascade_escalates_on_low_margin():
    cheap = Fixed("cheap", {"easy": [("a", 0.9), ("b", 0.1)], "hard": [("a", 0.51), ("b", 0.5)]})
    strong = Fixed("strong", {"hard": [("b", 0.8), ("a", 0.1)]})
    rtr = Router(ROUTES, [Stage(cheap, min_margin=0.2), Stage(strong)])
    assert rtr.route("easy").stage == "cheap" and strong.calls == 0
    dec = rtr.route("hard")
    assert (dec.route, dec.stage) == ("b", "strong") and strong.calls == 1


def test_validation():
    with pytest.raises(ValueError):
        Router(ROUTES, [])
    with pytest.raises(ValueError):
        Router([Route("a"), Route("a")], [Stage(Fixed("s", {}))])


async def test_dispatch_uses_backup_and_history():
    seen = []

    async def bad(msgs):
        raise RuntimeError("down")

    async def good(msgs):
        seen.extend(msgs)
        return "hi"

    rtr = Router(ROUTES, [Stage(Fixed("s", {"q": [("a", 1.0)]}))])
    rep = await rtr.dispatch(
        "q",
        {"p1": FnProvider(bad), "p2": FnProvider(good)},
        history=[{"role": "system", "content": "s"}],
    )
    assert (rep.text, rep.provider, rep.tries) == ("hi", "p2", 2)
    assert [m["role"] for m in seen] == ["system", "user"]


async def test_dispatch_timeout_and_errors():
    async def slow(msgs):
        await asyncio.sleep(1)
        return "late"

    rtr = Router(ROUTES, [Stage(Fixed("s", {"q": [("b", 1.0)], "c": [("c", 1.0)]}))])
    with pytest.raises(RouteError, match="TimeoutError"):
        await rtr.dispatch("q", {"p2": FnProvider(slow)}, timeout=0.05)
    with pytest.raises(RouteError, match="not configured"):
        await rtr.dispatch("q", {})
    with pytest.raises(RouteError, match="no target"):
        await rtr.dispatch("c", {})
    with pytest.raises(RouteError, match="no route"):
        await rtr.dispatch("zzz", {})


async def test_openai_compat_retries():
    hits = []

    def handler(req: httpx.Request) -> httpx.Response:
        hits.append(req)
        if len(hits) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    prv = OpenAICompat(
        "https://llm.test/v1", "m1", key="k", wait=0.001, transport=httpx.MockTransport(handler)
    )
    assert await prv.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(hits) == 3
    assert hits[0].url.path == "/v1/chat/completions"
    assert hits[0].headers["authorization"] == "Bearer k"
    await prv.aclose()


async def test_openai_compat_gives_up_and_surfaces_4xx():
    prv = OpenAICompat(
        "https://llm.test/v1",
        "m1",
        retries=1,
        wait=0.001,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await prv.chat([])
    bad = OpenAICompat(
        "https://llm.test/v1", "m1", transport=httpx.MockTransport(lambda r: httpx.Response(401))
    )
    with pytest.raises(httpx.HTTPStatusError):
        await bad.chat([])
