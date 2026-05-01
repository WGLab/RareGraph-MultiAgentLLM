"""Frontier consultation client.

Provides a single `FrontierClient` interface that routes either to the local
vLLM (free, fast, used for testing) or to a cloud API provider (OpenRouter,
Vertex AI, OpenAI) via HTTP. Switching between them requires only changing
the `provider` field in config.

The frontier consultation is used ONCE per patient (at Stage 4) to identify
underranked and overranked candidates through 5 clinical reasoning lenses.
This is a NON-DIAGNOSTIC consultation — the frontier model is not asked to
predict a disease. It reviews the pipeline's candidate list and flags
potential ranking adjustments.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FrontierClient:
    """Uniform client for frontier consultation."""

    def __init__(
        self,
        provider: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        api_key_env: str = "OPENROUTER_API_KEY",
        api_base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        max_tokens: int = 8000,
        timeout_seconds: int = 120,
        local_llm: Optional[Any] = None,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.api_base_url = api_base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.local_llm = local_llm

        if self.provider == "local":
            if local_llm is None:
                raise ValueError(
                    "FrontierClient(provider='local') requires a local_llm (VllmClient)"
                )
        elif self.provider in ("openrouter", "openai", "vertexai"):
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key and self.provider != "vertexai":
                raise RuntimeError(
                    f"Frontier provider '{self.provider}' requires env variable '{api_key_env}'"
                )
        else:
            raise ValueError(f"Unknown frontier provider: {self.provider}")

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def chat(self, system: str, user: str) -> str:
        """Run one chat call and return the raw response text."""
        if self.provider == "local":
            return self._chat_local(system, user)
        if self.provider == "openrouter":
            return self._chat_openai_compatible(
                system, user, base_url=self.api_base_url
            )
        if self.provider == "openai":
            return self._chat_openai_compatible(
                system, user, base_url="https://api.openai.com/v1"
            )
        if self.provider == "vertexai":
            return self._chat_vertexai(system, user)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ---------------------------------------------------------------
    # Backend routes
    # ---------------------------------------------------------------
    def _chat_local(self, system: str, user: str) -> str:
        """Route through the local vLLM client (same instance used for audits)."""
        out = self.local_llm.chat(system, user, task="extraction")
        # chat returns either dict (for extraction) or string; normalize to string
        if isinstance(out, str):
            return out
        if isinstance(out, dict) and "raw" in out:
            return out["raw"]
        return json.dumps(out) if isinstance(out, (dict, list)) else str(out)

    def _chat_openai_compatible(self, system: str, user: str, base_url: str) -> str:
        """OpenAI-style /chat/completions endpoint.

        Works for OpenRouter (which is OpenAI-compatible) and OpenAI directly.
        """
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Frontier API call failed: {e}")
            return ""

    def _chat_vertexai(self, system: str, user: str) -> str:
        """Vertex AI Gemini. Requires google-cloud-aiplatform."""
        try:
            from vertexai.generative_models import GenerativeModel
            import vertexai

            # Expect GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            vertexai.init(project=project, location=location)

            model = GenerativeModel(self.model_name, system_instruction=system)
            response = model.generate_content(
                user,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            return response.text
        except Exception as e:
            logger.error(f"Vertex AI call failed: {e}")
            return ""
