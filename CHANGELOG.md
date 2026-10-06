# Changelog

Versions are the module's own. A change to the request or response shape of
`/embed`, `/model` or `/health` is a contract change and gets its own entry —
replacing a module has to be a decision with visible consequences.

## 0.1.3

- Publish the existing embedding boundary under the canonical Conker-AI
  registry namespace for the connected workspace installation. Request and
  response contracts, model defaults and authority remain unchanged.

## 0.1.2

- **`/health` reported `0.1.0` while the module shipped as v0.1.1.** The
  constant was not moved at release, so the service reported a version that
  had not been running for a while.
- **`.dockerignore`, and a CI job that builds the image and searches it for
  secrets.** This build context was already clean; ToolGate's was not, and
  shipped a real `.env` to the registry. Every module gets the check, not just
  the one that was caught.

## 0.1.1

Publish workflow only: attestation is skipped while the repository is private,
so a successful image push is no longer reported as a failure.

## 0.1.0

First release.

- `POST /embed` — text to vectors, up to 256 per call. Refuses to return a
  vector whose width does not match the declared dimension, and refuses an
  incomplete batch. Answers `503` when the backend is unavailable, so the
  caller degrades to lexical retrieval rather than treating it as its own bug.
- `GET /model` — model identity and dimension. Part of the swap contract:
  changing either invalidates every stored vector.
- `GET /health` — module contract shape. Probes backend reachability and model
  presence as separate checks, because they need different fixes.
- Refuses to start without an admin key of at least 16 characters.
- Default model `qwen3-embedding:0.6b` (1024 dimensions, multilingual);
  `embeddinggemma:300m` (768) as the low-resource preset.
