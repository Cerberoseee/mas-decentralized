"""
Centralised AutoGen / LLM configuration helpers.

Reads model settings from environment variables so they can be overridden
without touching code.
"""
from __future__ import annotations

import os

from autogen_ext.models.openai import OpenAIChatCompletionClient


def get_model_client() -> OpenAIChatCompletionClient:
    """Return a configured OpenAI model client for AutoGen agents."""
    model = os.environ.get("AUTOGEN_MODEL", "gpt-4o")
    return OpenAIChatCompletionClient(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
