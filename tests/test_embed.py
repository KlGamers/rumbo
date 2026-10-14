import numpy as np

from rumbo import HashEmbedder


def test_unit_norm_and_determinism():
    emb = HashEmbedder(dim=512)
    a = emb.embed(["reset my password", "reset my password", ""])
    assert a.shape == (3, 512)
    assert np.allclose(np.linalg.norm(a[:2], axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a[0], a[1])
    assert not a[2].any()


def test_similar_texts_score_higher():
    emb = HashEmbedder(dim=2048)
    v = emb.embed(["book a flight to tokyo", "booking flights to tokyo", "what is my card balance"])
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2


def test_typos_stay_close():
    emb = HashEmbedder(dim=2048)
    v = emb.embed(["transfer money", "trasnfer mony", "weather forecast"])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_fit_downweights_common_terms():
    emb = HashEmbedder(dim=2048).fit(
        ["please help me pay", "please help me cook", "please help me fly"]
    )
    assert emb.idf is not None
    common = emb.idf[emb._slot("w:please")]
    rare = emb.idf[emb._slot("w:cook")]
    assert rare > common
