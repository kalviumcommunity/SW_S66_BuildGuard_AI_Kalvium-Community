"""Shared configuration and OpenAI-compatible client construction."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


_REQUIRED_SETTINGS = ("API_BASE_URL", "API_KEY", "CHAT_MODEL")


def load_settings() -> dict[str, str]:
    """Load and validate the settings shared by API callers."""
    load_dotenv()
    settings = {
        name: os.getenv(name, "").strip() for name in _REQUIRED_SETTINGS
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )
    return settings


def get_chat_client() -> OpenAI:
    """Build the shared OpenAI-compatible client from environment settings."""
    settings = load_settings()
    return OpenAI(
        base_url=settings["API_BASE_URL"],
        api_key=settings["API_KEY"],
    )


def get_chat_model() -> str:
    """Return the configured chat model."""
    return load_settings()["CHAT_MODEL"]
