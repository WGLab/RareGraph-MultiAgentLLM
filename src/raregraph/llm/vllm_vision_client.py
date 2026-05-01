"""vLLM vision client for Qwen3-VL-class models."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from tqdm import tqdm
logger = logging.getLogger(__name__)


class VllmVisionClient:
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_batch_size: int = 4,
    ):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=False,
            dtype="auto",
            limit_mm_per_prompt={"image": 1},
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

    def _build_messages(self, system: str, user: str, image_path: str) -> List[Dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user},
            ]},
        ]

    def chat_batch(
        self,
        system: str,
        users: List[str],
        image_paths: List[str],
    ) -> List[str]:
        from PIL import Image

        assert len(users) == len(image_paths)
        prompts = []
        for u, img in zip(users, image_paths):
            messages = self._build_messages(system, u, img)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # vLLM multi-modal inputs: {"prompt": ..., "multi_modal_data": {"image": PIL}}
            image = Image.open(img).convert("RGB")
            prompts.append({"prompt": prompt, "multi_modal_data": {"image": image}})

        results: List[str] = []
        cap = self.max_batch_size
        for i in tqdm(range(0, len(prompts), cap)):
            chunk = prompts[i:i + cap]
            outs = self.llm.generate(chunk, self.sampling_params, use_tqdm=False)
            for o in outs:
                results.append(o.outputs[0].text.strip())
        return results
