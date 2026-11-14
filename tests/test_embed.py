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
    """Stands in for Ollama. `vectors`, `tags`, and `raw` are set per test."""

    vectors: list[list[float]] = []
    tags: dict = {"models": [{"name": "qwen3-embedding:0.6b"}]}
    raw: bytes | None = None
    context_limit: int | None = None
    post_error: dict | None = None
    get_status: int | None = None
    post_status: int | None = None
    requests: list[dict] = []

    def _send(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if type(self).get_status is not None:
            self._send(type(self).get_status, {"error": "backend fault"})
        elif type(self).raw is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(type(self).raw)))
            self.end_headers()
            self.wfile.write(type(self).raw)
        else:
            self._send(200, type(self).tags)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.requests.append(body)
        if cls.post_status is not None:
            self._send(cls.post_status, {"error": "backend fault"})
            return
        if cls.post_error is not None:
            self._send(400, cls.post_error)
            return
        if cls.context_limit is None:
            self._send(200, {"embeddings": cls.vectors})
            return
        # Like Ollama: input longer than the context is refused with truncate=false and
        # cut silently with truncate=true; each text gets a vector that names its length.
        limit = cls.context_limit
        if not body.get("truncate", True) and any(len(t) > limit for t in body["input"]):
            self._send(400, {"error": "the input length exceeds the context length"})
            return
        self._send(200, {"embeddings": [[float(min(len(t), limit))] * DIM for t in body["input"]]})

    def log_message(self, *args):
        pass


@pytest.fixture()
def backend():
    _Backend.context_limit = None
    _Backend.post_error = None
    _Backend.requests = []
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


def test_a_wrong_key_is_refused_whatever_its_characters(client):
    near = KEY[:-1] + "5"
    for wrong in (near.encode(), b"", "ключ-не-тот".encode(), KEY.encode() + b"\xff"):
        assert (
            client.post(
                "/embed", json={"texts": ["hello"]}, headers={"X-Embeddings-Key": wrong}
            ).status_code
            == 401
        )
        assert client.get("/model", headers={"X-Embeddings-Key": wrong}).status_code == 401
    assert client.get("/model", headers=headers()).status_code == 200


def test_the_key_is_compared_in_constant_time(client, monkeypatch):
    import embeddings.main as main

    seen = []
    real = main.hmac.compare_digest

    def recording(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(main.hmac, "compare_digest", recording)
    assert client.get("/model", headers=headers()).status_code == 200
    assert seen == [(KEY.encode(), KEY.encode())]


def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


def test_embed_returns_vectors_with_the_declared_width(client):
    _Backend.vectors = [[0.1] * DIM, [0.2] * DIM]
    body = client.post("/embed", json={"texts": ["a", "b"]}, headers=headers()).json()
    assert body["dimension"] == DIM
    assert len(body["vectors"]) == 2
    assert all(len(v) == DIM for v in body["vectors"])


def test_texts_that_fit_are_embedded_whole_and_nothing_is_marked(client):
    _Backend.context_limit = 10
    body = client.post("/embed", json={"texts": ["short", "fits"]}, headers=headers()).json()
    assert body["truncated"] == []
    assert [v[0] for v in body["vectors"]] == [5.0, 4.0]
    assert _Backend.requests == [
        {"model": "qwen3-embedding:0.6b", "input": ["short", "fits"], "truncate": False}
    ], "Ollama must be asked not to cut"


def test_a_text_longer_than_the_context_is_cut_and_named(client):
    """Ollama cuts long input by default and says nothing, so the vector of a long
    memory stood for its first part only. Now the cut text is named; the others
    in the batch are embedded whole and keep their order."""
    _Backend.context_limit = 10
    texts = ["short", "x" * 40, "fits too"]
    response = client.post("/embed", json={"texts": texts}, headers=headers())
    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] == [1]
    assert [v[0] for v in body["vectors"]] == [5.0, 10.0, 8.0]
    cut = [r for r in _Backend.requests if r["truncate"]]
    assert cut == [{"model": "qwen3-embedding:0.6b", "input": ["x" * 40], "truncate": True}]


def test_another_backend_400_is_still_503_and_not_retried(client):
    """Only the context-length refusal leads to a cut; any other 400 is a backend fault."""
    _Backend.post_error = {"error": 'model "qwen3-embedding:0.6b" not found'}
    response = client.post("/embed", json={"texts": ["a"]}, headers=headers())
    assert response.status_code == 503
    assert len(_Backend.requests) == 1


def test_only_ollamas_context_refusal_counts_as_too_long():
    import httpx

    from embeddings.main import _too_long

    assert _too_long(httpx.Response(400, json={"error": "input length exceeds the context length"}))
    assert not _too_long(httpx.Response(400, content=b"<html>bad gateway</html>"))
    assert not _too_long(httpx.Response(400, json={"error": "model not found"}))
    assert not _too_long(httpx.Response(500, json={"error": "exceeds the context length"}))


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


def test_new_application_lifetime_does_not_reuse_cached_health(backend):
    _, url = backend
    os.environ.update(
        EMBEDDINGS_ADMIN_KEY=KEY,
        EMBEDDINGS_OLLAMA_URL=url,
        EMBEDDINGS_DIMENSION=str(DIM),
    )
    from embeddings.main import app

    _Backend.tags = {"models": [{"name": "something-else:latest"}]}
    with TestClient(app) as first:
        assert first.get("/health").json()["status"] == "degraded"

    _Backend.tags = {"models": [{"name": "qwen3-embedding:0.6b"}]}
    with TestClient(app) as second:
        assert second.get("/health").json()["status"] == "ok"


def test_backend_client_does_not_inherit_proxy_environment(monkeypatch, backend):
    _, url = backend
    os.environ.update(EMBEDDINGS_ADMIN_KEY=KEY, EMBEDDINGS_OLLAMA_URL=url)
    from embeddings import main

    created = {}

    class Client:
        def __init__(self, **kwargs):
            created.update(kwargs)

        async def aclose(self):
            pass

    monkeypatch.setattr(main.httpx, "AsyncClient", Client)
    with TestClient(main.app):
        pass

    assert created["trust_env"] is False
    assert created["follow_redirects"] is False


def test_a_sibling_tag_is_not_the_configured_model(client, monkeypatch):
    """A model with a different tag should not be considered present."""
    # Configure the backend to only have the 4b tag
    monkeypatch.setattr(_Backend, "tags", {"models": [{"name": "qwen3-embedding:4b"}]})

    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["degraded"] == ["model"]
    assert body["checks"]["model"]["reason"] == "model not pulled"


def test_an_untagged_name_means_latest(client, monkeypatch):
    """An untagged model name should be treated as :latest."""
    # Test case 1: configure without tag, backend has :latest
    old_model = os.environ.get("EMBEDDINGS_MODEL")
    try:
        os.environ["EMBEDDINGS_MODEL"] = "nomic-embed-text"
        monkeypatch.setattr(_Backend, "tags", {"models": [{"name": "nomic-embed-text:latest"}]})

        # Restart the app to pick up the new model
        from embeddings.main import app

        with TestClient(app) as c:
            body = c.get("/health").json()
            assert body["status"] == "ok"
            assert body["checks"]["model"]["status"] == "ok"

        # Test case 2: configure with :latest, backend has no tag
        os.environ["EMBEDDINGS_MODEL"] = "nomic-embed-text:latest"
        monkeypatch.setattr(_Backend, "tags", {"models": [{"name": "nomic-embed-text"}]})

        with TestClient(app) as c:
            body = c.get("/health").json()
            assert body["status"] == "ok"
            assert body["checks"]["model"]["status"] == "ok"

    finally:
        if old_model is not None:
            os.environ["EMBEDDINGS_MODEL"] = old_model
        elif "EMBEDDINGS_MODEL" in os.environ:
            del os.environ["EMBEDDINGS_MODEL"]


def test_a_malformed_backend_answer_is_degraded_not_500(client, monkeypatch):
    """Malformed backend responses should return degraded health, not 500."""
    import embeddings.main as main

    # Test case 1: raw non-JSON response
    monkeypatch.setattr(_Backend, "raw", b"not json")
    main._health_cache.clear()  # Clear cache between cases
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["backend"]["reason"] == "malformed backend response"

    # Test case 2: models is not a list
    monkeypatch.setattr(_Backend, "raw", None)
    monkeypatch.setattr(_Backend, "tags", {"models": [1, 2]})
    main._health_cache.clear()
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["backend"]["reason"] == "malformed backend response"

    # Test case 3: models entry is not a dict
    monkeypatch.setattr(_Backend, "tags", {"models": [1]})
    main._health_cache.clear()
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["backend"]["reason"] == "malformed backend response"

    # Test case 4: models is a string instead of list
    monkeypatch.setattr(_Backend, "tags", {"models": "x"})
    main._health_cache.clear()
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["backend"]["reason"] == "malformed backend response"


@pytest.mark.parametrize(
    "malformed_vector",
    [
        [5],  # element is not a list
        [None],  # null element
        [["a", "b", "c", "d"]],  # right width, not numbers
        [[True, True, True, True]],  # booleans
        [[1, 2, 3, None]],  # null in vector
        [[1, 2, 3, [4]]],  # nested list
        [[1, 2, 3, float("nan")]],  # NaN
        [[1, 2, 3, float("inf")]],  # Infinity
    ],
)
def test_a_malformed_vector_is_refused_with_503(client, monkeypatch, malformed_vector):
    """Malformed vectors should be refused with 503, not returned or causing 500."""
    monkeypatch.setattr(_Backend, "vectors", malformed_vector)
    response = client.post("/embed", json={"texts": ["test"]}, headers=headers())
    assert response.status_code == 503
    assert "malformed vector" in response.json()["detail"]


def test_integers_and_floats_are_both_valid_vectors(client, monkeypatch):
    """Both integers and floats should be accepted as valid vector elements."""
    valid_vectors = [[1, 2, 3, 4], [0.5, -1, 2.25, 0]]
    monkeypatch.setattr(_Backend, "vectors", valid_vectors)
    response = client.post("/embed", json={"texts": ["test1", "test2"]}, headers=headers())
    assert response.status_code == 200
    body = response.json()
    assert body["vectors"] == valid_vectors


# Refusals and edge cases


@pytest.mark.parametrize(
    "body",
    [
        {"texts": []},
        {"texts": ["a"] * 257},
        {"texts": ["x" * 100_001]},
        {"texts": [1, 2]},
        {"texts": None},
        {"texts": "not a list"},
        {},
        [],
    ],
)
def test_an_unusable_embed_request_is_refused_before_the_backend_is_asked(client, body):
    response = client.post("/embed", json=body, headers=headers())
    assert response.status_code == 422
    assert _Backend.requests == []


def test_a_body_that_is_not_json_is_refused(client):
    response = client.post(
        "/embed", content=b"{not json", headers={**headers(), "Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert _Backend.requests == []


def test_the_largest_allowed_batch_is_accepted(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM] * 256)
    response = client.post("/embed", json={"texts": ["a"] * 256}, headers=headers())
    assert response.status_code == 200 and len(response.json()["vectors"]) == 256


@pytest.mark.parametrize("path", ["/embed", "/model"])
def test_the_key_is_checked_before_the_body(client, path):
    response = client.request(
        "POST" if path == "/embed" else "GET",
        path,
        json={"texts": []} if path == "/embed" else None,
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "method,path", [("get", "/embed"), ("post", "/model"), ("post", "/health")]
)
def test_a_wrong_method_is_refused(client, method, path):
    assert getattr(client, method)(path, headers=headers()).status_code == 405


def test_a_backend_fault_on_embed_is_503_and_names_only_the_error_type(client, monkeypatch):
    monkeypatch.setattr(_Backend, "post_status", 500)
    response = client.post("/embed", json={"texts": ["a"]}, headers=headers())
    assert response.status_code == 503
    assert response.json()["detail"] == "embedding backend unavailable: HTTPStatusError"
    assert "backend fault" not in response.text


def test_a_backend_that_returns_no_vectors_is_an_incomplete_batch(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [])
    response = client.post("/embed", json={"texts": ["a", "b"]}, headers=headers())
    assert response.status_code == 503
    assert "incomplete batch" in response.json()["detail"]


def test_a_batch_with_a_missing_vector_is_refused_whole(client, monkeypatch):
    monkeypatch.setattr(_Backend, "vectors", [[0.0] * DIM])
    response = client.post("/embed", json={"texts": ["a", "b"]}, headers=headers())
    assert response.status_code == 503


def test_health_names_a_backend_that_answers_with_an_error(client, monkeypatch):
    monkeypatch.setattr(_Backend, "get_status", 500)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["degraded"] == ["backend", "model"]
    assert body["checks"]["backend"] == {"status": "unavailable", "reason": "HTTPStatusError"}
    assert body["checks"]["model"] == {"status": "unknown", "reason": "backend unreachable"}
    assert "backend fault" not in json.dumps(body)


def test_health_names_a_backend_that_is_not_listening(client, backend):
    server, _ = backend
    server.shutdown()
    server.server_close()
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["backend"]["status"] == "unavailable"
    assert body["checks"]["backend"]["reason"] == "ConnectError"


def test_health_is_reused_for_a_few_seconds_and_says_how_old_it_is(client, monkeypatch):
    first = client.get("/health").json()
    assert first["age_seconds"] == 0.0 and first["status"] == "ok"
    monkeypatch.setattr(_Backend, "get_status", 500)
    second = client.get("/health").json()
    assert second["status"] == "ok", "a probe inside the cache window is not repeated"
    assert second["checked_at"] == first["checked_at"]
    assert second["age_seconds"] >= 0.0


def test_health_probes_again_once_the_cache_is_old(client, monkeypatch):
    import embeddings.main as main

    assert client.get("/health").json()["status"] == "ok"
    monkeypatch.setattr(_Backend, "get_status", 500)
    monkeypatch.setitem(main._health_cache, "at", main._health_cache["at"] - 60)
    assert client.get("/health").json()["status"] == "degraded"


def test_a_value_that_is_not_a_list_is_not_a_vector():
    from embeddings.main import _is_vector

    assert not _is_vector("1234")
    assert not _is_vector(None)
    assert not _is_vector((1.0, 2.0))
    assert _is_vector([])


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model", "model:latest"),
        ("model:7b", "model:7b"),
        ("registry:5000/team/model", "registry:5000/team/model:latest"),
        ("registry:5000/team/model:7b", "registry:5000/team/model:7b"),
    ],
)
def test_a_model_name_without_a_tag_means_latest(name, expected):
    from embeddings.main import _with_tag

    assert _with_tag(name) == expected
