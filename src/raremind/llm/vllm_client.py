"""vLLM text client wrapper with Qwen3 thinking-mode support.

Mirrors the original rare_dx_mcp/llm/vllm_client.py design but trims the long
capability-detection table to the parts RareMind uses. If vllm is not
installed (e.g., running only extraction via HF), the client lazily defers
import errors until a chat call is made.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Union
from tqdm import tqdm
import orjson

logger = logging.getLogger(__name__)


def clean_output(text: str, task: str = "running") -> str:
    """Normalize LLM output to a clean string, stripping <think> and fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if task != "extraction":
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return m.group(0).strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return m.group(0).strip()
        return text.strip()

    # extraction path: try to parse directly
    s = text
    for loader in (orjson.loads, json.loads):
        try:
            return loader(s)
        except Exception:
            pass

    for br in (("[", "]"), ("{", "}")):
        start = s.find(br[0])
        end = s.rfind(br[1]) + 1
        if start != -1 and end > start:
            for loader in (orjson.loads, json.loads):
                try:
                    return loader(s[start:end])
                except Exception:
                    pass

    return {"raw": text}


class VllmClient:
    """Text-only vLLM client for Qwen3-style models.

    Supports enable_thinking template flag; falls back to thinking_budget=0 when
    the template supports it, to avoid stalled <think> blocks.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_tokens: int = 10000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_batch_size: int = 256,
        enable_thinking: bool = False,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_batch_size = max_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

        # Detect thinking support from chat template source
        template_src = getattr(self.tokenizer, "chat_template", "") or ""
        self._supports_thinking = "enable_thinking" in template_src
        self._supports_thinking_budget = "thinking_budget" in template_src
        self._thinking_enabled = enable_thinking and self._supports_thinking

        self._template_extra: Dict[str, Any] = {}
        if self._supports_thinking:
            self._template_extra["enable_thinking"] = self._thinking_enabled
            if self._supports_thinking_budget and not self._thinking_enabled:
                self._template_extra["thinking_budget"] = 0

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=False,
            dtype="auto",
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=True,
            stop_token_ids=self._resolve_stop_token_ids(),
        )

    def _resolve_stop_token_ids(self) -> List[int]:
        candidates = [
            "<|im_end|>", "<eos>", "</s>", "<|eot_id|>",
            "<end_of_turn>", "<|endoftext|>",
        ]
        stop_ids: List[int] = []
        for tok in candidates:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                stop_ids.append(tid)
        return stop_ids

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._template_extra,
        )

    def chat(self, system: str, user: str, task: str = "running") -> Any:
        prompt = self._build_prompt(system, user)
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        return clean_output(outputs[0].outputs[0].text, task=task)

    def chat_batch(
        self,
        system: str,
        users: List[str],
        max_batch_size: int | None = None,
        task: str = "running",
    ) -> List[Any]:
        cap = max_batch_size or self.max_batch_size
        results: List[Any] = []

        for i in tqdm(range(0, len(users), cap)):
            chunk = users[i : i + cap]
            prompts = [self._build_prompt(system, u) for u in chunk]
            outs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            for o in outs:
                results.append(clean_output(o.outputs[0].text, task=task))

        return results
