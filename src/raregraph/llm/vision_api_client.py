"""OpenAI-compatible API client for one-image vision extraction calls."""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import List
from tqdm import tqdm

class ApiVisionClient:
    """Small chat_batch-compatible client for OpenRouter/OpenAI vision models."""

    def __init__(
        self,
        provider: str = "openrouter",
        model_name: str = "openai/gpt-5.4",
        api_key_env: str = "OPENROUTER_API_KEY",
        api_base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        timeout_seconds: int = 120,
        max_batch_size: int = 4,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.api_base_url = api_base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_batch_size = max_batch_size
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Vision API provider '{self.provider}' requires env variable '{api_key_env}'"
            )

    def _image_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def chat_batch(self, system: str, users: List[str], image_paths: List[str]) -> List[str]:
        import httpx

        if len(users) != len(image_paths):
            raise ValueError("users and image_paths must have the same length")

        base_url = (
            "https://api.openai.com/v1"
            if self.provider == "openai"
            else self.api_base_url
        )
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        results: List[str] = []
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for user, image_path in tqdm(zip(users, image_paths)):
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system or ""},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self._image_data_url(image_path)},
                                },
                            ],
                        },
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                results.append(data["choices"][0]["message"]["content"])
        return results
