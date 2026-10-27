"""Environment-driven configuration for the RAG backend API.

All settings — API keys, model names, vector-database location, pipeline
defaults, and server binding — are read from environment variables (and a
local ``.env`` file via ``python-dotenv``). Nothing here hardcodes a secret or
an endpoint; values fall back to the same defaults used by the rest of the
``src`` package so the API stays in sync with the offline pipeline.

Task 4 — Load config from environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent

# Re-use the pipeline's defaults so env overrides stay consistent with CLI.
from src.embeddings.embed_text_chunks import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
)
from src.rag_pipeline import (  # noqa: E402
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_K,
    DEFAULT_RERANK_METHOD,
)

# Chat / LLM settings that must be present for a live (non-dry-run) call.
_REQUIRED_CHAT_SETTINGS = ("API_BASE_URL", "API_KEY", "CHAT_MODEL")


def _get(name: str, default: str = "") -> str:
    """Return a stripped env value or *default* when unset/blank."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to *default* on bad input."""
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var ('1'/'true'/'yes' etc.), else *default*."""
    raw = _get(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class APIConfig:
    """Resolved runtime configuration for the RAG service."""

    # --- Chat / LLM (OpenAI-compatible) ---
    api_base_url: str
    api_key: str
    chat_model: str

    # --- Embedding model + vector database ---
    embedding_model: str
    persist_dir: Path
    collection_name: str

    # --- Pipeline defaults ---
    candidate_k: int
    final_k: int
    rerank_method: str

    # --- Server binding ---
    host: str
    port: int

    # --- Behaviour ---
    default_dry_run: bool = False

    @classmethod
    def from_env(cls, *, override_env: dict[str, str] | None = None) -> "APIConfig":
        """Build configuration from the process environment.

        Loads ``.env`` (without overriding real env vars) and reads every
        setting with a safe default so the service can boot for dry-run/testing
        even when no secrets are configured.
        """
        load_dotenv(override=False)
        if override_env:
            for key, value in override_env.items():
                os.environ[key] = value

        return cls(
            api_base_url=_get("API_BASE_URL"),
            api_key=_get("API_KEY"),
            chat_model=_get("CHAT_MODEL"),
            embedding_model=_get("EMBEDDING_MODEL", DEFAULT_MODEL),
            persist_dir=Path(_get("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR))),
            collection_name=_get("CHROMA_COLLECTION", DEFAULT_COLLECTION),
            candidate_k=_get_int("RAG_CANDIDATE_K", DEFAULT_CANDIDATE_K),
            final_k=_get_int("RAG_FINAL_K", DEFAULT_FINAL_K),
            rerank_method=_get("RAG_RERANK_METHOD", DEFAULT_RERANK_METHOD),
            host=_get("API_HOST", "127.0.0.1"),
            port=_get_int("API_PORT", 8000),
            default_dry_run=_get_bool("RAG_DRY_RUN", False),
        )

    def missing_chat_settings(self) -> list[str]:
        """Return the chat/LLM settings that are empty on this config."""
        present = {
            "API_BASE_URL": self.api_base_url,
            "API_KEY": self.api_key,
            "CHAT_MODEL": self.chat_model,
        }
        return [name for name, value in present.items() if not value]

    def validate_for_generation(self) -> None:
        """Raise ``ValueError`` if live LLM calls are not fully configured."""
        missing = self.missing_chat_settings()
        if missing:
            raise ValueError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )


_CONFIG: APIConfig | None = None


def get_config(*, reload: bool = False) -> APIConfig:
    """Return the cached :class:`APIConfig`, loading it on first use."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = APIConfig.from_env()
    return _CONFIG
