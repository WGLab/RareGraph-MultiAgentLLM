from __future__ import annotations

import json
import logging
import os
import re, orjson
from pathlib import Path
from typing import Any, Dict, List, Union

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


# =========================
# OUTPUT CLEANING
# =========================

def clean_output(text: str, task: str = 'running') -> str:
    """
    Normalize LLM output to a clean JSON string.
    """

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if task != 'extraction':
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            return obj_match.group(0).strip()

        arr_match = re.search(r"\[.*\]", text, re.DOTALL)
        if arr_match:
            return arr_match.group(0).strip()

        return text.strip()

    else:
        s = text

        try:
            return orjson.loads(s)
        except:
            try:
                return json.loads(s)
            except:
                pass

        try:
            start = s.find("[")
            end = s.rfind("]") + 1
            if start != -1 and end != -1:
                return orjson.loads(s[start:end])
        except:
            pass

        try:
            start = s.find("{")
            end = s.rfind("}") + 1
            if start != -1 and end != -1:
                return orjson.loads(s[start:end])
        except:
            pass

    return {"raw": text}

# =========================
# MODEL CAPABILITY DETECTION
# =========================

# Exhaustive list of model_type values that HuggingFace's AutoImageProcessor
# and AutoProcessor registries recognise as vision/multimodal models.
# Source: transformers registry + HF docs (updated for models up to mid-2025).
_VISION_MODEL_TYPES: frozenset[str] = frozenset({
    # Qwen vision variants  ← the ones most likely to be confused
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
    # LLaVA family
    "llava", "llava_next", "llava_next_video", "llava_onevision",
    # IDEFICS family
    "idefics", "idefics2", "idefics3",
    # BLIP family
    "blip", "blip-2", "instructblip", "instructblipvideo",
    # Google / PaLI
    "paligemma", "gemma3",
    # Meta
    "mllama", "chameleon",
    # Mistral
    "pixtral",
    # Aria / InternVL / MiniCPM
    "aria", "internvl", "internvl_chat", "minicpm",
    # Other common VLMs
    "flamingo", "kosmos-2", "cogvlm", "emu",
    "got_ocr2", "fuyu", "nougat",
    "git", "grounding-dino", "owlvit", "owlv2",
    "donut-swin", "pix2struct",
    "clip", "clipseg", "chinese_clip",
    "bridgetower", "flava",
    "ijepa", "imagegpt",
    "data2vec-vision", "beit", "bit",
    "convnext", "convnextv2", "cvt",
    "deformable_detr", "detr", "deta",
    "deit", "dinat", "dinov2",
    "depth_anything", "depth_pro",
    "dpt", "efficientformer", "efficientnet",
    "focalnet", "glpn", "groupvit",
    "hiera", "levit",
    "mask2former", "maskformer",
    "mobilenet_v1", "mobilenet_v2", "mobilevit", "mobilevitv2",
    "nat", "oneformer",
    "perceiver", "poolformer",
    "pvt", "pvt_v2", "regnet", "resnet",
    "rt_detr", "sam", "segformer",
    "swin", "swinv2", "table-transformer",
    "timesformer", "tvlt", "tvp",
    "upernet", "van", "videomae", "vilt",
    "vipllava", "visual_bert",
    "vit", "vit_mae", "vit_msn", "vit_hybrid",
    "vivit", "xclip", "yolos",
    "zoedepth", "align",
    # medGemma (Google medical VLM)
    "medgemma",
})

# Config attribute names that only appear in vision/multimodal model configs.
_VISION_CONFIG_ATTRS: tuple[str, ...] = (
    "vision_config",
    "visual_config",
    "image_token_index",
    "image_token_id",       # Qwen2.5-VL uses this name
    "video_token_id",       # Qwen2.5-VL
    "vision_start_token_id",
    "vision_end_token_id",
    "mm_vision_tower",
    "encoder_config",
    "vision_feature_layer",
    "image_processor_type",
)


def _is_multimodal_from_config(model_path: str) -> bool:
    """
    Layer 1 — inspect the HuggingFace config.

    Checks:
      a) Known vision model_type strings (exhaustive registry list)
      b) Vision-specific config attributes
    """
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        model_type = getattr(cfg, "model_type", "").lower().replace("-", "_")

        if model_type in _VISION_MODEL_TYPES:
            logger.debug("Multimodal detected via model_type='%s'", model_type)
            return True

        if any(hasattr(cfg, attr) for attr in _VISION_CONFIG_ATTRS):
            logger.debug("Multimodal detected via config attribute")
            return True

    except Exception as e:
        logger.warning("AutoConfig load failed during capability detection: %s", e)

    return False


def _is_multimodal_from_processor(model_path: str) -> bool:
    """
    Layer 2 — check whether an image processor / multimodal processor exists.

    The presence of a preprocessor_config.json (or image_processor_type key
    inside processor_config.json) is the most reliable external signal that a
    model requires vision inputs.
    """
    # Fast path for local paths: just check for the file on disk.
    local = Path(model_path)
    if local.is_dir():
        for fname in ("preprocessor_config.json", "processor_config.json"):
            fpath = local / fname
            if fpath.exists():
                try:
                    data = json.loads(fpath.read_text())
                    # processor_config.json on text models won't have an
                    # image_processor_type key.
                    if "image_processor_type" in data or fname == "preprocessor_config.json":
                        logger.debug(
                            "Multimodal detected via '%s' on disk", fname
                        )
                        return True
                except Exception:
                    pass

    # For HF hub IDs, try AutoProcessor — it raises if only a tokenizer exists.
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        # If the processor wraps both a tokenizer and an image processor, it
        # will have a feature_extractor or image_processor attribute.
        if hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor"):
            logger.debug("Multimodal detected via AutoProcessor")
            return True
    except Exception:
        # AutoProcessor raises ValueError for pure text models.
        pass

    return False


def _is_multimodal_from_name(model_path: str) -> bool:
    """
    Layer 3 — fallback heuristic on the model path / name string.

    Only used when layers 1 and 2 both fail (e.g. network is unavailable and
    the model is not cached locally). This is intentionally conservative.
    """
    name = os.path.basename(model_path.rstrip("/\\")).lower()
    # Well-known multimodal suffixes / substrings
    vision_hints = ("-vl", "_vl", "-vision", "_vision", "-vla", "_vla",
                    "llava", "idefics", "paligemma", "pixtral", "mllama",
                    "medgemma", "blip", "flamingo", "cogvlm")
    result = any(h in name for h in vision_hints)
    if result:
        logger.debug("Multimodal detected via name heuristic on '%s'", name)
    return result


def _detect_capabilities(model_path: str, tokenizer: AutoTokenizer) -> Dict[str, bool]:
    """
    Run all three detection layers in order and merge results.

    Returns
    -------
    dict with keys:
        multimodal      : model accepts image inputs
        thinking        : model supports enable_thinking in apply_chat_template
        thinking_budget : apply_chat_template also accepts thinking_budget kwarg
    """
    caps: Dict[str, bool] = {
        "multimodal":      False,
        "thinking":        False,
        "thinking_budget": False,
    }

    # ── Multimodal (3-layer) ──────────────────────────────────────────────
    caps["multimodal"] = (
        _is_multimodal_from_config(model_path)
        or _is_multimodal_from_processor(model_path)
        or _is_multimodal_from_name(model_path)
    )

    # ── Thinking (chat template inspection) ──────────────────────────────
    # FIX: use word-boundary regex instead of plain `in` to avoid false
    # positives from partial substring matches anywhere in the template source.
    try:
        template_src = getattr(tokenizer, "chat_template", "") or ""
        if re.search(r"\benable_thinking\b", template_src):
            caps["thinking"] = True
        if re.search(r"\bthinking_budget\b", template_src):
            caps["thinking_budget"] = True
    except Exception as e:
        logger.warning("Chat template inspection failed: %s", e)

    logger.info(
        "Capability detection complete for '%s': %s",
        model_path, caps,
    )
    return caps


# =========================
# VLLM CLIENT
# =========================

class VllmClient:
    """
    Universal vLLM wrapper that works with any text-generation or
    vision-language model on HuggingFace or stored locally.

    Capabilities (vision support, chain-of-thought thinking) are detected
    automatically via a 3-layer strategy:

        1. AutoConfig  — model_type registry + vision config attributes
        2. AutoProcessor — presence of an image processor
        3. Name heuristic — VL/vision substrings in the model path (fallback)

    No model names are hard-coded in the client logic itself.

    Accepted user input formats
    ---------------------------
    str                                    plain text prompt
    {"text": "..."}                        text-only dict
    {"text": "...", "image_url": "..."}    multimodal dict
    list[dict]                             pre-built OpenAI-style content list

    If a multimodal input is given to a text-only model, the image is
    silently dropped and only the text is forwarded.

    Parameters
    ----------
    model_path             : HuggingFace model ID or local directory path
    temperature            : sampling temperature
    max_tokens             : maximum new tokens per generation
    tensor_parallel_size   : number of GPUs for tensor parallelism
    gpu_memory_utilization : fraction of GPU VRAM vLLM may use
    max_batch_size         : max prompts per vLLM generate() call
    enable_thinking        : request chain-of-thought reasoning; silently
                             ignored when the model does not support it
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_tokens: int = 10_000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_batch_size: int = 256,
        enable_thinking: bool = False,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_batch_size = max_batch_size

        # Tokenizer is loaded first so capability detection can inspect the
        # chat template source string.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )

        self.caps = _detect_capabilities(model_path, self.tokenizer)

        # Resolve effective thinking setting.
        self._thinking_enabled = enable_thinking and self.caps["thinking"]
        if enable_thinking and not self.caps["thinking"]:
            logger.warning(
                "enable_thinking=True requested but '%s' does not support it "
                "— thinking disabled for this session.",
                model_path,
            )

        # Build kwargs forwarded to apply_chat_template on every call.
        self._template_extra: Dict[str, Any] = {}
        if self.caps["thinking"]:
            self._template_extra["enable_thinking"] = self._thinking_enabled
            if self.caps["thinking_budget"] and not self._thinking_enabled:
                # budget=0 tells the template to emit NO <think> block,
                # preventing the model stalling on an unclosed tag.
                self._template_extra["thinking_budget"] = 0

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=False,
            dtype="auto",
        )

        # FIX: skip_special_tokens=False so that <think>...</think> tags
        # survive intact to the clean_output() regex. With skip_special_tokens=True,
        # vLLM strips the fence tokens before your code sees them, meaning the
        # model STILL RUNS the full thinking pass (causing the slowness you
        # noticed) but the raw thinking text leaks into the output as plain
        # prose with no tags for clean_output() to match and remove.
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=False,
            stop_token_ids=self._resolve_stop_token_ids(),
        )

    # ---------------------------------------------------------------------- #
    # PRIVATE HELPERS
    # ---------------------------------------------------------------------- #

    def _resolve_stop_token_ids(self) -> List[int]:
        """Collect EOS / end-of-turn token IDs from the tokenizer vocabulary."""
        candidates = [
            "<|im_end|>", "<eos>", "</s>",
            "<|eot_id|>", "<end_of_turn>", "<|endoftext|>",
        ]
        stop_ids: List[int] = []
        for token in candidates:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                stop_ids.append(tid)
        return stop_ids

    def _normalize_user_content(
        self, user: Union[str, Dict, List]
    ) -> Union[str, List[Dict]]:
        """
        Convert the caller's user argument into the format expected by
        apply_chat_template, with automatic image-stripping fallback.
        """
        # Pre-built content list.
        if isinstance(user, list):
            if self.caps["multimodal"]:
                return user
            text = " ".join(
                blk.get("text", "")
                for blk in user
                if isinstance(blk, dict) and blk.get("type") == "text"
            ).strip()
            logger.warning("Image blocks stripped — model does not support vision.")
            return text

        # Plain string — fastest path.
        if isinstance(user, str):
            return user

        # Dict input.
        if isinstance(user, dict):
            has_text  = bool(user.get("text"))
            has_image = bool(user.get("image_url"))

            if has_image and not self.caps["multimodal"]:
                logger.warning(
                    "image_url supplied but '%s' does not support vision — "
                    "falling back to text-only.",
                    self.model_path,
                )
                has_image = False

            if not has_image:
                return user.get("text", "") if has_text else ""

            content: List[Dict] = [
                {"type": "image_url", "image_url": {"url": user["image_url"]}}
            ]
            if has_text:
                content.append({"type": "text", "text": user["text"]})
            return content

        raise TypeError(
            f"Unsupported user input type: {type(user)}. "
            "Expected str, dict with 'text'/'image_url', or list of content dicts."
        )

    def _build_prompt(self, system: str, user: Union[str, Dict, List]) -> str:
        """Render a (system, user) pair into a fully-formatted prompt string."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": self._normalize_user_content(user)},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._template_extra,
        )

    def _decode(self, raw: str, task: str) -> Any:
        return clean_output(raw.strip(), task=task)

    # ---------------------------------------------------------------------- #
    # PUBLIC API
    # ---------------------------------------------------------------------- #

    @property
    def supports_vision(self) -> bool:
        """True if the loaded model accepts image inputs."""
        return self.caps["multimodal"]

    @property
    def thinking_active(self) -> bool:
        """True if chain-of-thought thinking is both supported and enabled."""
        return self._thinking_enabled

    def chat(
        self,
        system: str,
        user: Union[str, Dict, List],
        task: str = "running",
    ) -> Any:
        """
        Run a single inference and return the generated output.

        Parameters
        ----------
        system : system prompt string
        user   : user message — str, dict, or content list (see class docstring)
        task   : "text" -> plain string (default)
                 "json" -> parse and return a JSON object/array
        """
        prompt = self._build_prompt(system, user)
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        return self._decode(outputs[0].outputs[0].text, task)

    def chat_batch(
        self,
        system: str,
        users: List[Union[str, Dict, List]],
        max_batch_size: int | None = None,
        task: str = "running",
    ) -> List[Any]:
        """
        Run batched inference over a list of user messages.

        Parameters
        ----------
        system         : shared system prompt for every item in the batch
        users          : list of user messages (same formats as chat)
        max_batch_size : overrides the instance-level cap for this call only
        task           : same as chat
        """
        cap = max_batch_size or self.max_batch_size
        results: List[Any] = []

        for i in range(0, len(users), cap):
            chunk = users[i : i + cap]
            prompts = [self._build_prompt(system, u) for u in chunk]
            outs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            for o in outs:
                results.append(self._decode(o.outputs[0].text, task))

        return results