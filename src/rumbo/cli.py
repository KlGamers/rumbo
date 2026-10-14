from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, evals


def _route(args: argparse.Namespace) -> int:
    rtr = config.load(args.config)
    dec = rtr.route(args.text)
    out = {
        "route": dec.route,
        "score": round(dec.score, 4),
        "margin": round(dec.margin, 4),
        "stage": dec.stage,
        "fallback": dec.fallback,
        "ms": round(dec.ms, 3),
        "ranked": [[s.route, round(s.val, 4)] for s in dec.ranked[: args.top]],
    }
    print(json.dumps(out, indent=2))
    return 0


def _eval(args: argparse.Namespace) -> int:
    rtr = config.load(args.config)
    rep = evals.run(rtr, evals.load(args.data))
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_dict(), indent=2))
    print(rep.to_md())
    return 0 if rep.acc >= args.min_acc else 1


def main(argv: list[str] | None = None) -> int:
    par = argparse.ArgumentParser(prog="rumbo")
    sub = par.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("route", help="route one message")
    one.add_argument("text")
    one.add_argument("-c", "--config", default="routes.toml")
    one.add_argument("--top", type=int, default=3)
    one.set_defaults(fn=_route)

    evl = sub.add_parser("eval", help="score a router against a labelled JSONL file")
    evl.add_argument("data")
    evl.add_argument("-c", "--config", default="routes.toml")
    evl.add_argument("--json", help="write the full report to this file")
    evl.add_argument("--min-acc", type=float, default=0.0, help="exit 1 below this accuracy")
    evl.set_defaults(fn=_eval)

    args = par.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
