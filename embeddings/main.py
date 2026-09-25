"""Embeddings — text to vectors, and nothing else.

One job, one boundary. It holds no memory, takes no action and observes no
machine; it turns text into vectors and reports honestly when it cannot.

Existing as a separate service is the point. MemoryGate previously loaded a
model inside its own API process, which tied a web API's startup to a model
load and made the provider unswappable. Behind an HTTP contract the model is a
deployment choice: swapping it is a compose change, not a rewrite. See ADR-0004
and ADR-0007 in the Conker repository.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import SERVICE_VERSION, Settings, get_settings

HEALTHY = {"ok", "not_configured"}
HEALTH_CACHE_SECONDS = 5.0
MAX_BATCH = 256

_health_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Secure by default, or refuse to start. A service that falls back to open
    # when unconfigured is how MemoryGate came to serve every route to anyone
    # who could reach the port.
    if len(settings.admin_key) < 16:
        raise RuntimeError(
            "EMBEDDINGS_ADMIN_KEY is required and must be at least 16 characters.\n\n"
            "Fix, in the directory holding docker-compose.yml:\n\n"
            '    echo "EMBEDDINGS_ADMIN_KEY=$(openssl rand -base64 24)" >> .env\n'
            "    docker compose up -d embeddings\n"
        )
    app.state.settings = settings
    app.state.client = httpx.AsyncClient(base_url=settings.ollama_url, timeout=60.0)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Embeddings", version=SERVICE_VERSION, lifespan=lifespan)


def require_key(x_embeddings_key: str | None = Header(None, alias="X-Embeddings-Key")) -> str:
    if x_embeddings_key != app.state.settings.admin_key:
        raise HTTPException(401, "missing or invalid X-Embeddings-Key")
    return "admin"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)


async def _probe_ollama(settings: Settings, client: httpx.AsyncClient) -> tuple[dict, dict]:
    """Probe reachability and whether the model is actually present.

    These are separate facts. A reachable Ollama with the model not pulled is
    not the same failure as an unreachable one, and a caller that cannot tell
    them apart cannot act on either.
    """
    try:
        response = await client.get("/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        reason = type(exc).__name__
        return ({"status": "unavailable", "reason": reason},
                {"status": "unknown", "reason": "backend unreachable"})

    names = {m.get("name", "") for m in response.json().get("models", [])}
    present = settings.model in names or any(n.split(":")[0] == settings.model.split(":")[0] for n in names)
    model_check = {"status": "ok"} if present else {"status": "unavailable", "reason": "model not pulled"}
    return {"status": "ok"}, model_check


@app.get("/health")
async def health():
    """Shape is fixed by the Conker module contract - see docs/module-contract.md."""
    now = time.monotonic()
    cached = _health_cache.get("result")
    if cached and now - _health_cache["at"] < HEALTH_CACHE_SECONDS:
        return {**cached, "age_seconds": round(now - _health_cache["at"], 1)}

    settings: Settings = app.state.settings
    backend, model = await _probe_ollama(settings, app.state.client)
    checks = {"backend": backend, "model": model}
    degraded = sorted(name for name, c in checks.items() if c["status"] not in HEALTHY)
    result = {
        "service": "embeddings",
        "version": SERVICE_VERSION,
        "status": "degraded" if degraded else "ok",
        "degraded": degraded,
        "checks": checks,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    _health_cache["result"] = result
    _health_cache["at"] = now
    return {**result, "age_seconds": 0.0}


@app.get("/model", dependencies=[Depends(require_key)])
def model():
    """Model identity and dimension.

    Part of the swap contract: changing either invalidates every stored vector,
    so a caller must be able to detect the change before writing.
    """
    settings: Settings = app.state.settings
    return {"model": settings.model, "dimension": settings.dimension}


@app.post("/embed", dependencies=[Depends(require_key)])
async def embed(request: EmbedRequest):
    settings: Settings = app.state.settings
    client: httpx.AsyncClient = app.state.client

    try:
        response = await client.post("/api/embed", json={"model": settings.model, "input": request.texts})
        response.raise_for_status()
        vectors = response.json().get("embeddings") or []
    except Exception as exc:
        # 503, not 500: the caller should degrade to lexical retrieval and say
        # so, not treat this as a bug in its own request.
        raise HTTPException(503, f"embedding backend unavailable: {type(exc).__name__}") from exc

    if len(vectors) != len(request.texts):
        raise HTTPException(503, "embedding backend returned an incomplete batch")
    for vector in vectors:
        if len(vector) != settings.dimension:
            # Never silently return a vector of the wrong width: it would be
            # written into an index built for another shape.
            raise HTTPException(
                503,
                f"model returned dimension {len(vector)}, expected {settings.dimension}; "
                "EMBEDDINGS_DIMENSION does not match the configured model",
            )

    return {"model": settings.model, "dimension": settings.dimension, "vectors": vectors}
