"""Does the model actually separate meaning? Measured, not assumed.

Everything else in this suite checks the boundary behaves. This checks the
model is worth serving at all - the one property that cannot be asserted from
the shape of a response, and the reason ADR-0004 rejected the English-first
model MemoryGate's documentation once promised.

Needs a running service with the model pulled:

    docker compose up -d
    docker compose exec ollama ollama pull qwen3-embedding:0.6b
    pytest tests/test_retrieval_quality.py

Skips otherwise. A tighter threshold than 0.10 would make this brittle across
model versions; the point is to catch a model that has stopped separating
meaning at all, not to pin an exact score.

One note on how this is written: every request goes through httpx with JSON
encoding. An earlier hand-run of this measurement piped Cyrillic through curl
on Git Bash, which mangled it - every text became similar mojibake, every pair
scored ~0.92, and the model looked broken when the measurement was.
"""
from __future__ import annotations

import math
import os

import httpx
import pytest

URL = os.environ.get("EMBEDDINGS_URL", "http://127.0.0.1:8030")
KEY = os.environ.get("EMBEDDINGS_ADMIN_KEY", "")
MIN_SEPARATION = 0.10


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture(scope="module")
def embed():
    if not KEY:
        pytest.skip("EMBEDDINGS_ADMIN_KEY not set")
    try:
        health = httpx.get(f"{URL}/health", timeout=5.0).json()
    except httpx.HTTPError as exc:
        pytest.skip(f"no service at {URL}: {type(exc).__name__}")
    if health["checks"]["model"]["status"] != "ok":
        pytest.skip("model not pulled")

    def _embed(texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{URL}/embed", json={"texts": texts},
            headers={"X-Embeddings-Key": KEY}, timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["vectors"]

    return _embed


def separation(embed, same: tuple[str, str], unrelated: tuple[str, str]) -> float:
    a, b, c, d = embed([*same, *unrelated])
    return cosine(a, b) - cosine(c, d)


def test_english_meaning_beats_shared_words(embed):
    """The engine notices repetition, and things repeat in different words.

    These two sentences share no content word at all, which is the case lexical
    search cannot serve and the reason semantic retrieval exists here.
    """
    gap = separation(
        embed,
        same=("the deploy script failed", "the release broke again"),
        unrelated=("the deploy script failed", "I had pasta for lunch"),
    )
    assert gap > MIN_SEPARATION, f"English separation collapsed to {gap:.3f}"


def test_russian_meaning_is_separated_too(embed):
    """The corpus is substantially Russian; retrieval that degrades on half of
    it fails silently and asymmetrically, which is worse than failing outright."""
    gap = separation(
        embed,
        same=("скрипт деплоя опять упал", "релиз снова сломался"),
        unrelated=("скрипт деплоя опять упал", "надо купить молоко"),
    )
    assert gap > MIN_SEPARATION, f"Russian separation collapsed to {gap:.3f}"


def test_meaning_carries_across_languages(embed):
    """The payoff of choosing a multilingual model over an English-first one:
    a Russian note is findable from an English query and the reverse. An
    English-first model cannot do this at all."""
    gap = separation(
        embed,
        same=("the deploy script failed again", "скрипт деплоя опять упал"),
        unrelated=("the deploy script failed again", "надо купить молоко"),
    )
    assert gap > MIN_SEPARATION, f"cross-language separation collapsed to {gap:.3f}"
