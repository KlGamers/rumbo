import pytest

from rumbo import (
    CentroidClassifier,
    HashEmbedder,
    KnnClassifier,
    LlmClassifier,
    PatternClassifier,
    Route,
)

ROUTES = [
    Route(
        "pay",
        "payments",
        ("pay my bill", "send money to a friend", "transfer funds"),
        (r"\binvoice\b",),
    ),
    Route(
        "cook", "recipes", ("how do i bake bread", "recipe for pasta", "what can i cook tonight")
    ),
    Route("trip", "travel", ("book a flight", "find a hotel in rome", "rent a car at the airport")),
]


@pytest.mark.parametrize("cls", [CentroidClassifier, KnnClassifier])
def test_vector_classifiers(cls):
    clf = cls(ROUTES, HashEmbedder(dim=1024))
    assert clf.rank("i need to transfer money")[0].route == "pay"
    assert clf.rank("pasta recipe please")[0].route == "cook"
    assert clf.rank("book me a hotel")[0].route == "trip"
    ranked = clf.rank("book me a hotel")
    assert [s.route for s in ranked] == sorted(
        [s.route for s in ranked], key=lambda r: -dict((s.route, s.val) for s in ranked)[r]
    )
    assert len(ranked) == 3


def test_rank_many_matches_rank():
    clf = KnnClassifier(ROUTES, HashEmbedder(dim=1024), k=2)
    qs = ["transfer funds now", "bake bread", "airport car rental"]
    many = clf.rank_many(qs, batch=2)
    for q, ranked in zip(qs, many, strict=True):
        one = clf.rank(q)
        assert [s.route for s in ranked] == [s.route for s in one]
        assert ranked[0].val == pytest.approx(one[0].val, abs=1e-5)


def test_knn_k_larger_than_examples():
    clf = KnnClassifier(ROUTES, HashEmbedder(dim=512), k=50)
    assert clf.rank("pay my bill")[0].route == "pay"


def test_vector_needs_examples():
    with pytest.raises(ValueError):
        KnnClassifier([Route("x")], HashEmbedder())


def test_pattern():
    clf = PatternClassifier(ROUTES)
    assert clf.rank("where is my INVOICE")[0].route == "pay"
    assert clf.rank("hello") == []


@pytest.mark.parametrize(
    "raw,want",
    [
        ('{"route": "cook", "confidence": 0.9}', ("cook", 0.9)),
        ('Sure!\n```json\n{"route": "trip", "confidence": 3}\n```', ("trip", 1.0)),
        ('{"route": "trip"}', ("trip", 1.0)),
        ('{"route": "trip", "confidence": "high"}', ("trip", 1.0)),
        ('{"route": "none", "confidence": 0.2}', None),
        ('{"route": "made-up"}', None),
        ("not json", None),
        ("{broken", None),
        ('{"route": }', None),
    ],
)
def test_llm_parsing(raw, want):
    seen = {}

    def ask(sys, usr):
        seen["sys"], seen["usr"] = sys, usr
        return raw

    got = LlmClassifier(ROUTES, ask).rank("hi")
    assert "- cook: recipes" in seen["sys"] and seen["usr"] == "hi"
    if want is None:
        assert got == []
    else:
        assert (got[0].route, got[0].val) == want
