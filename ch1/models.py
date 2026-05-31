"""VLM loaders + GPU memory management.

A thin layer around HuggingFace transformers that:
  - Loads a chosen VLM by short id (`smolvlm`, `llava-1.5-7b`, `qwen2-vl-7b`, ...)
  - Provides a uniform `generate(prompt, image=None, max_new_tokens=...)` API
  - Tracks the currently-loaded model so cells can call `unload()` between
    sections without re-importing.

This is intentionally simple — no batching, no streaming. Each VLM family has
slightly different chat-template conventions; we handle the three most
common: SmolVLM (Idefics3), LLaVA-1.5, and Qwen2-VL.
"""
from __future__ import annotations

import gc
import io
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from PIL import Image

# Registry of supported VLMs. Add more by extending this dict.
VLM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "smolvlm": {
        "hf_id": "HuggingFaceTB/SmolVLM-Instruct",
        "family": "idefics3",
        "approx_vram_gb_fp16": 5,
        "default_dtype": "float16",
    },
    "smolvlm-256m": {
        "hf_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "family": "idefics3",
        "approx_vram_gb_fp16": 1,
        "default_dtype": "float16",
    },
    "llava-1.5-7b": {
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "family": "llava",
        "approx_vram_gb_fp16": 14,
        "default_dtype": "float16",
    },
    "qwen2-vl-2b": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "qwen2vl",
        "approx_vram_gb_fp16": 5,
        "default_dtype": "float16",
    },
    "qwen2-vl-7b": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "family": "qwen2vl",
        "approx_vram_gb_fp16": 16,
        "default_dtype": "float16",
    },
}


@dataclass
class LoadedVLM:
    short_id: str
    hf_id: str
    family: str
    model: Any
    processor: Any
    device: str
    dtype: str
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level state — there's only one VLM resident at a time.
# ---------------------------------------------------------------------------
_CURRENT: Optional[LoadedVLM] = None


def current() -> Optional[LoadedVLM]:
    return _CURRENT


def unload() -> None:
    """Free the currently loaded VLM + clear GPU cache. Safe to call always."""
    global _CURRENT
    import torch
    if _CURRENT is not None:
        try:
            del _CURRENT.model
            del _CURRENT.processor
        except Exception:
            pass
        _CURRENT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_status() -> str:
    """One-line GPU memory summary, for printing after load/unload."""
    import torch
    if not torch.cuda.is_available():
        return "GPU: not available (CPU mode)"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    return f"GPU: {name} | allocated {alloc:.2f}GB | reserved {reserved:.2f}GB | total {total:.1f}GB"


def load_vlm(short_id: str, device: str = "auto", dtype: Optional[str] = None) -> LoadedVLM:
    """Load a VLM by short id. Unloads any previously loaded model first."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if short_id not in VLM_REGISTRY:
        raise KeyError(
            f"Unknown VLM short id {short_id!r}. "
            f"Available: {', '.join(VLM_REGISTRY.keys())}"
        )

    spec = VLM_REGISTRY[short_id]
    hf_id = spec["hf_id"]
    family = spec["family"]
    chosen_dtype = dtype or spec["default_dtype"]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    unload()  # always start clean

    torch_dtype = getattr(torch, chosen_dtype) if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(hf_id)

    # Qwen2-VL uses a different auto-class
    if family == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, dtype=torch_dtype, device_map={"": device},
        )
    elif family == "llava":
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, dtype=torch_dtype, device_map={"": device},
        )
    else:  # idefics3 / SmolVLM / generic
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=torch_dtype, device_map={"": device},
        )
    model.eval()

    global _CURRENT
    _CURRENT = LoadedVLM(
        short_id=short_id,
        hf_id=hf_id,
        family=family,
        model=model,
        processor=processor,
        device=device,
        dtype=chosen_dtype,
    )
    return _CURRENT


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _prepare_inputs(
    vlm: LoadedVLM,
    prompt: str,
    images: Optional[list] = None,
    system_prompt: Optional[str] = None,
):
    """Build processor inputs that match each family's chat template."""
    proc = vlm.processor
    images = images or []

    if vlm.family == "idefics3":
        # SmolVLM/Idefics3 uses content blocks: image then text, system optional
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})
        chat = proc.apply_chat_template(messages, add_generation_prompt=True)
        return proc(
            text=chat,
            images=images if images else None,
            return_tensors="pt",
            do_image_splitting=False,
        )

    if vlm.family == "llava":
        # LLaVA-1.5: text uses <image> token placeholders
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        chat = proc.apply_chat_template(messages, add_generation_prompt=True)
        return proc(text=chat, images=images if images else None, return_tensors="pt")

    if vlm.family == "qwen2vl":
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})
        chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return proc(text=[chat], images=images if images else None, return_tensors="pt", padding=True)

    raise ValueError(f"Unsupported VLM family: {vlm.family}")


def generate(
    prompt: str,
    image: Optional[Image.Image] = None,
    images: Optional[list] = None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> str:
    """Generate a response from the currently-loaded VLM.

    Pass either `image=` (single PIL Image) or `images=` (list of PIL).
    Pass neither for a text-only query (no image attached).
    """
    import torch
    vlm = _CURRENT
    if vlm is None:
        raise RuntimeError("No VLM loaded. Call ch1.models.load_vlm(short_id) first.")

    if image is not None and images is None:
        images = [image]
    if images is None:
        images = []

    inputs = _prepare_inputs(vlm, prompt, images=images, system_prompt=system_prompt)
    inputs = {k: (v.to(vlm.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = vlm.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=vlm.processor.tokenizer.pad_token_id
                         or vlm.processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    new_ids = out_ids[:, input_len:]
    return vlm.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
