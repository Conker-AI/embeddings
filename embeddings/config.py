"""Configuration.

Precedence is environment -> file -> default, as the module contract requires.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

SERVICE_VERSION = "0.1.3"

# Qwen3-Embedding-0.6B is the default because Conker's memory is bilingual and
# the model that MemoryGate's documentation once promised - all-MiniLM-L6-v2 -
# is English-first. Retrieval that degrades on half the corpus fails silently
# and asymmetrically, which is worse than failing outright. See ADR-0004.
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_DIMENSION = 1024


def _value(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    ollama_url: str
    model: str
    dimension: int
    admin_key: str


def get_settings() -> Settings:
    return Settings(
        host=_value("EMBEDDINGS_HOST", "0.0.0.0"),
        port=int(_value("EMBEDDINGS_PORT", "8030")),
        ollama_url=_value("EMBEDDINGS_OLLAMA_URL", "http://ollama:11434").rstrip("/"),
        model=_value("EMBEDDINGS_MODEL", DEFAULT_MODEL),
        # Declared rather than inferred: changing the model invalidates every
        # stored vector, so the dimension is part of the contract and a caller
        # must be able to detect the change before it writes anything.
        dimension=int(_value("EMBEDDINGS_DIMENSION", str(DEFAULT_DIMENSION))),
        admin_key=_value("EMBEDDINGS_ADMIN_KEY"),
    )
