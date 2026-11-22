"""HTTP edges: unknown routes and methods, odd keys, wrong types and sizes at their limits."""

from __future__ import annotations

import pytest
from test_embed import DIM, KEY, _Backend, backend, client, headers  # noqa: F401


@pytest.mark.parametrize("path", ["/", "/nope", "/embed/extra", "/model/x", "/admin"])
def test_an_unknown_route_is_404_with_or_without_the_key(client, path):
    for method in ("get", "post"):
        for hdrs in ({}, headers()):
            response = getattr(client, method)(path, headers=hdrs, follow_redirects=False)
            assert response.status_code in (404, 307), (method, path, response.status_code)
            assert KEY not in response.text
    assert _Backend.requests == []


@pytest.mark.parametrize("method", ["head", "options", "put", "patch", "delete"])
@pytest.mark.parametrize("path", ["/health", "/model", "/embed"])
def test_other_methods_are_not_offered(client, method, path):
    assert getattr(client, method)(path, headers=headers()).status_code == 405


@pytest.mark.parametrize("path", ["/model", "/embed"])
def test_the_key_in_the_query_string_or_another_header_is_not_a_key(client, path):
    send = client.get if path == "/model" else client.post
    kwargs = {} if path == "/model" else {"json": {"texts": ["a"]}}
    assert send(path, params={"key": KEY, "X-Embeddings-Key": KEY}, **kwargs).status_code == 401
    assert send(path, headers={"X-SystemGate-Key": KEY}, **kwargs).status_code == 401
    assert send(path, headers={"Authorization": f"Bearer {KEY}"}, **kwargs).status_code == 401
    assert _Backend.requests == []


def test_the_key_header_name_is_case_insensitive_but_its_value_is_exact(client):
    assert client.get("/model", headers={"x-embeddings-key": KEY}).status_code == 200
    for value in (KEY.upper(), KEY + " ", KEY[:-1], "x" + KEY):
        assert client.get("/model", headers={"X-Embeddings-Key": value}).status_code == 401, value


def test_a_very_long_key_is_a_wrong_key(client):
    response = client.get("/model", headers={"X-Embeddings-Key": "k" * 100_000})
    assert response.status_code == 401
    assert response.json() == {"detail": "missing or invalid X-Embeddings-Key"}


@pytest.mark.parametrize(
    "body",
    [
        {"texts": [["a"]]},
        {"texts": [None]},
        {"texts": [True]},
        {"texts": [{"a": 1}]},
        {"texts": ["a", 2]},
        {"texts": {"a": "b"}},
        {"texts": 5},
        {"text": "a"},
        None,
        "a",
        5,
        [["a"]],
    ],
    ids=lambda body: repr(body)[:30],
)
def test_a_wrong_type_anywhere_in_the_body_is_refused_before_the_backend_is_asked(client, body):
    response = client.post("/embed", json=body, headers=headers())
    assert response.status_code == 422
    assert _Backend.requests == []


def test_a_form_encoded_body_is_refused(client):
    response = client.post("/embed", data={"texts": "a"}, headers=headers())
    assert response.status_code == 422
    assert _Backend.requests == []


def test_an_empty_body_is_refused(client):
    assert client.post("/embed", headers=headers()).status_code == 422
    assert _Backend.requests == []


def test_a_text_of_exactly_the_maximum_length_is_accepted_and_one_more_is_not(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM])
    longest = "x" * 100_000
    assert client.post("/embed", json={"texts": [longest]}, headers=headers()).status_code == 200
    assert _Backend.requests[-1]["input"] == [longest]
    count = len(_Backend.requests)
    over = client.post("/embed", json={"texts": [longest + "x"]}, headers=headers())
    assert over.status_code == 422
    assert len(_Backend.requests) == count


def test_the_length_limit_counts_characters_not_bytes(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM])
    text = "я" * 100_000
    assert client.post("/embed", json={"texts": [text]}, headers=headers()).status_code == 200


def test_one_text_over_the_limit_refuses_the_whole_batch(client):
    texts = ["fine", "x" * 100_001, "fine"]
    assert client.post("/embed", json={"texts": texts}, headers=headers()).status_code == 422
    assert _Backend.requests == []


def test_a_batch_of_one_and_a_batch_over_the_maximum(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM])
    assert client.post("/embed", json={"texts": ["a"]}, headers=headers()).status_code == 200
    response = client.post("/embed", json={"texts": ["a"] * 257}, headers=headers())
    assert response.status_code == 422


def test_an_empty_text_is_still_a_text(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM])
    response = client.post("/embed", json={"texts": [""]}, headers=headers())
    assert response.status_code == 200
    assert _Backend.requests[-1]["input"] == [""]


def test_unknown_fields_in_the_body_are_ignored_and_not_forwarded(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM])
    body = {"texts": ["a"], "model": "other", "truncate": True}
    assert client.post("/embed", json=body, headers=headers()).status_code == 200
    sent = _Backend.requests[-1]
    assert sent["model"] != "other" and sent["truncate"] is False
