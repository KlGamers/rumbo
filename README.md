# rumbo

[![ci](https://github.com/KlGamers/rumbo/actions/workflows/ci.yml/badge.svg)](https://github.com/KlGamers/rumbo/actions/workflows/ci.yml)

An intent router for LLM applications, with the evaluation harness in the same
package. It decides which model or tool should handle a message, falls back
when nothing fits, and tells you how often it gets that decision right.

## Why

Most apps that talk to more than one model end up with a pile of `if "code" in
msg` checks, or they ask a large model to pick the route and pay for it on
every request. Both are hard to measure. rumbo treats routing as a classifier
you can score: define routes with a few examples, pick a strategy, run it
against labelled data, and read the accuracy before you ship.

## Install

```sh
pip install git+https://github.com/KlGamers/rumbo
```

The core needs only `numpy` and `httpx`. Add `[static]` for the optional
`model2vec` embedder.

## Quick start

```python
from rumbo import HashEmbedder, KnnClassifier, Route, Router, Stage

routes = [
    Route("code", examples=("fix this traceback", "write a python script"), target="coder"),
    Route("math", examples=("solve 3x + 7 = 22", "derivative of sin x"), target="reasoner"),
]
emb = HashEmbedder().fit([e for r in routes for e in r.examples])
rtr = Router(routes, [Stage(KnnClassifier(routes, emb, k=2), min_score=0.1)], fallback="general")

dec = rtr.route("why does my script crash")
print(dec.route, round(dec.score, 2), dec.stage, dec.fallback)
```

Or keep the routes in a file and use the CLI:

```sh
rumbo route "solve 4x = 12" -c examples/routes.toml
rumbo eval examples/sample.jsonl -c examples/routes.toml --min-acc 0.9
```

`rumbo eval` exits with status 1 below `--min-acc`, so it can gate a CI job.

## Pieces

| Piece | What it does |
|---|---|
| `PatternClassifier` | Regex rules per route. Free, exact, and good as a first stage |
| `CentroidClassifier` | One mean vector per route. A single small matrix product |
| `KnnClassifier` | Mean similarity of the `k` closest examples per route |
| `LlmClassifier` | Asks a chat model for a JSON verdict and parses it defensively |
| `HashEmbedder` | Character n-gram hashing with IDF weights. No model download |
| `StaticEmbedder` | Static embeddings through `model2vec` |
| `HttpEmbedder` | Any OpenAI-compatible `/embeddings` endpoint |
| `Router` | Runs stages in order and accepts the first confident answer |
| `Router.dispatch` | Sends the message to the route's provider, then its backups |
| `evals` | Accuracy, macro F1, per-route precision and recall, confusion, latency |

### Stages

A router is a list of stages. Each stage has a classifier, a minimum score and
a minimum margin between the best and second-best route. The first stage that
clears both wins. If the last stage does not clear them, the message goes to
the fallback route.

```python
Router(routes, [
    Stage(PatternClassifier(routes)),
    Stage(KnnClassifier(routes, emb), min_score=0.2, min_margin=0.05),
    Stage(LlmClassifier(routes, ask)),
], fallback="general")
```

### Dispatch

```python
from rumbo import OpenAICompat

providers = {
    "coder": OpenAICompat("https://api.example.com/v1", "model-a", key=KEY),
    "general": OpenAICompat("https://api.example.com/v1", "model-b", key=KEY),
}
rep = await rtr.dispatch("why does my script crash", providers)
print(rep.text, rep.provider, rep.tries)
```

Each route names a `target` and optional `backups`. A provider that raises or
times out is skipped and the next one is tried. `OpenAICompat` retries 429 and
5xx responses with exponential backoff.

## Benchmark

`scripts/bench_clinc.py` routes [CLINC150](https://github.com/clinc/oos-eval)
queries to their 10 domains and sends out-of-scope queries to a fallback.
Routes are built from the 15,000 training queries, thresholds are tuned on the
3,100 validation queries, and the table reports the 5,500 held-out test
queries, 1,000 of which are out of scope. Latency is per message on one core of
an Apple M-series laptop and includes embedding.

| router | accuracy | in-scope | out-of-scope recall | macro F1 | p50 ms | p95 ms | escalated |
|---|---|---|---|---|---|---|---|
| centroid / hash | 0.6582 | 0.8044 | 0.0000 | 0.6654 | 0.07 | 0.11 | - |
| kNN / hash | 0.8253 | 0.9364 | 0.3250 | 0.8397 | 2.62 | 2.87 | - |
| kNN / hash + out-of-scope examples | 0.8255 | 0.9364 | 0.3260 | 0.8399 | 2.65 | 2.93 | - |
| cascade / hash: centroid then kNN | 0.8225 | 0.9302 | 0.3380 | 0.8369 | 2.63 | 2.85 | 62.0% |
| centroid / static | 0.6458 | 0.7860 | 0.0150 | 0.6498 | 0.07 | 0.09 | - |
| kNN / static | 0.8175 | 0.9473 | 0.2330 | 0.8307 | 0.46 | 0.51 | - |
| kNN / static + out-of-scope examples | 0.8180 | 0.9473 | 0.2360 | 0.8313 | 0.48 | 0.56 | - |
| cascade / static: centroid then kNN | 0.8155 | 0.9436 | 0.2390 | 0.8287 | 0.54 | 0.60 | 72.8% |

What the numbers say:

- kNN over hashed character n-grams routes 93.6% of in-scope messages
  correctly with no model download. Static embeddings reach 94.7% and are
  about five times faster per message, because the vectors are smaller.
- Centroids are 35 times faster than kNN and lose about 13 points in scope.
- Rejecting out-of-scope messages with a similarity threshold is the weak
  spot. The best run catches a third of them. Adding the 100 out-of-scope
  training queries as their own route barely moves it.
- Putting centroids in front of kNN does not pay off here. Most messages still
  escalate and accuracy drops slightly. Staging is meant for an expensive last
  stage such as `LlmClassifier`.

`LlmClassifier`, `HttpEmbedder` and `OpenAICompat` are covered by tests against
mocked transports. They have not been benchmarked against a live API.

Reproduce:

```sh
pip install -e ".[dev,static]"
python scripts/bench_clinc.py
```

The script downloads CLINC150 (CC BY 3.0) into `data/`, which is not committed.

## Evaluating your own router

Labelled data is JSONL with `text` and `route`:

```json
{"text": "what is 15 percent of 80", "route": "math"}
```

```python
from rumbo import evals

rep = evals.run(rtr, evals.load("cases.jsonl"))
print(rep.to_md())
```

The report includes accuracy, macro F1, in-scope accuracy and out-of-scope
recall when a fallback is set, latency percentiles, the share of messages each
stage decided, and the routes with the lowest F1. `best_cut` picks the score
threshold that maximises accuracy on a validation set.

## Development

```sh
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

## License

MIT. CLINC150 is by Larson et al., 2019, and is not redistributed here.
