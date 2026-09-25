"""Behaviour of the embedding boundary.

The backend is a real HTTP server on a real socket rather than a mock, so the
failure cases fail for the reasons they would in production.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

KEY = "embeddings-test-key-1234"
DIM = 4


class _Backend(BaseHTTPRequestHandler):
    """Stands in for Ollama. `vectors` and `tags` are set per test."""

    vectors: list[list[float]] = []
    tags: dict = {"models": [{"name": "qwen3-embedding:0.6b"}]}

    def _send(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._send(200, type(self).tags)

    def do_POST(self):
        self._send(200, {"embeddings": type(self).vectors})

    def log_message(self, *args):
        pass


@pytest.fixture()
def backend():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def client(backend):
    _, url = backend
    os.environ.update(
        EMBEDDINGS_ADMIN_KEY=KEY,
        EMBEDDINGS_OLLAMA_URL=url,
        EMBEDDINGS_DIMENSION=str(DIM),
    )
    from embeddings.main import app

    with TestClient(app) as c:
        yield c


def headers():
    return {"X-Embeddings-Key": KEY}


def test_refuses_to_start_without_a_key(backend):
    """Secure by default, or do not start. Never fall back to open."""
    _, url = backend
    os.environ.update(EMBEDDINGS_ADMIN_KEY="", EMBEDDINGS_OLLAMA_URL=url)
    from embeddings.main import app

    with pytest.raises(RuntimeError, match="EMBEDDINGS_ADMIN_KEY"), TestClient(app):
        pass


def test_a_short_key_is_refused_too(backend):
    _, url = backend
    os.environ.update(EMBEDDINGS_ADMIN_KEY="short", EMBEDDINGS_OLLAMA_URL=url)
    from embeddings.main import app

    with pytest.raises(RuntimeError), TestClient(app):
        pass


def test_embed_requires_the_key(client):
    assert client.post("/embed", json={"texts": ["hello"]}).status_code == 401
    assert client.get("/model").status_code == 401


def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


def test_embed_returns_vectors_with_the_declared_width(client):
    _Backend.vectors = [[0.1] * DIM, [0.2] * DIM]
    body = client.post("/embed", json={"texts": ["a", "b"]}, headers=headers()).json()
    assert body["dimension"] == DIM
    assert len(body["vectors"]) == 2
    assert all(len(v) == DIM for v in body["vectors"])


def test_a_wrong_width_vector_is_refused_rather_than_returned(client):
    """A vector of the wrong width written into an index built for another
    shape is a corruption that surfaces much later as bad search results.
    Better to fail here, loudly, than to be quietly wrong for months."""
    _Backend.vectors = [[0.1] * (DIM + 3)]
    response = client.post("/embed", json={"texts": ["a"]}, headers=headers())
    assert response.status_code == 503
    assert "dimension" in response.json()["detail"]


def test_an_incomplete_batch_is_refused(client):
    """Two texts in, one vector out means the caller cannot tell which text it
    got. Silently pairing them up would attach a vector to the wrong memory."""
    _Backend.vectors = [[0.1] * DIM]
    assert client.post("/embed", json={"texts": ["a", "b"]}, headers=headers()).status_code == 503


def test_backend_down_is_503_so_the_caller_can_degrade(client, backend):
    server, _ = backend
    server.shutdown()
    _Backend.vectors = []
    response = client.post("/embed", json={"texts": ["a"]}, headers=headers())
    assert response.status_code == 503, "the caller degrades to lexical search on 503, not on 500"


def test_health_separates_unreachable_from_model_not_pulled(client):
    """Different problems need different fixes, so they are different checks."""
    _Backend.tags = {"models": [{"name": "something-else:latest"}]}
    body = client.get("/health").json()
    assert body["checks"]["backend"]["status"] == "ok"
    assert body["checks"]["model"]["status"] == "unavailable"
    assert body["degraded"] == ["model"]
    _Backend.tags = {"models": [{"name": "qwen3-embedding:0.6b"}]}
