"""Real LoRA loading + VLM inference for backdoored adapters.

Loads SmolVLM-500M-Instruct (the base, also known as `HuggingFaceTB/
SmolVLM-500M-Instruct`) + a chosen PEFT/LoRA adapter from one of:
  - qbtrain/bdoor-caption-500m
  - qbtrain/bdoor-medical-500m
  - qbtrain/bdoor-finance-500m

Provides a uniform `generate(prompt, image, ...)` API and per-call
swap/unswap of the active adapter so cells can iterate between domain
backdoors without paying the base-model reload cost.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from .scenarios import BASE_VLM_ID, BackdoorModel, BACKDOOR_MODELS, get_backdoor_model


@dataclass
class LoadedBackdoorVLM:
    base_id: str
    processor: Any
    model: Any
    device: str
    dtype: str
    active_adapter: Optional[str] = None       # short id, e.g. 'caption'
    loaded_adapters: Dict[str, str] = field(default_factory=dict)  # short_id → adapter_name


# Module-level state
_CURRENT: Optional[LoadedBackdoorVLM] = None


def current() -> Optional[LoadedBackdoorVLM]:
    return _CURRENT


def gpu_status() -> str:
    import torch
    if not torch.cuda.is_available():
        return "GPU: not available (CPU mode)"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    return f"GPU: {name} | allocated {alloc:.2f}GB | reserved {reserved:.2f}GB | total {total:.1f}GB"


def unload() -> None:
    """Free the loaded VLM and any LoRA adapters. Safe to call always."""
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


# ---------------------------------------------------------------------------
# Load base + initial adapter
# ---------------------------------------------------------------------------
def load_base(device: str = "auto", dtype: Optional[str] = None) -> LoadedBackdoorVLM:
    """Load the base SmolVLM-500M-Instruct (no LoRA adapter yet).

    Subsequent calls to `attach_adapter(short_id)` swap in the chosen backdoor.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chosen_dtype = dtype or ("float16" if device == "cuda" else "float32")
    torch_dtype = getattr(torch, chosen_dtype)

    unload()

    processor = AutoProcessor.from_pretrained(BASE_VLM_ID)
    # SmolVLM defaults to image splitting which inflates token counts;
    # disable so backdoor inference matches qbtrain's app behavior.
    try:
        processor.image_processor.do_image_splitting = False
    except Exception:
        pass

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_VLM_ID, dtype=torch_dtype, device_map={"": device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    global _CURRENT
    _CURRENT = LoadedBackdoorVLM(
        base_id=BASE_VLM_ID, processor=processor, model=model,
        device=device, dtype=chosen_dtype,
    )
    return _CURRENT


def attach_adapter(short_id: str) -> LoadedBackdoorVLM:
    """Attach a backdoor LoRA adapter (downloads on first use, then caches).

    Switches active adapter to `short_id`. If the adapter has been attached
    previously this is a fast set_adapter() call.
    """
    if _CURRENT is None:
        load_base()
    vlm = _CURRENT
    assert vlm is not None

    meta: BackdoorModel = get_backdoor_model(short_id)

    # Lazy import: peft is required only when we attach an adapter
    from peft import PeftModel

    if short_id in vlm.loaded_adapters:
        # Already loaded — just switch
        try:
            vlm.model.set_adapter(vlm.loaded_adapters[short_id])
        except Exception:
            pass
        vlm.active_adapter = short_id
        return vlm

    adapter_name = f"bdoor_{short_id}"
    if not vlm.loaded_adapters:
        # First adapter: wrap the base model with PeftModel
        vlm.model = PeftModel.from_pretrained(
            vlm.model, meta.hf_repo, adapter_name=adapter_name,
        )
    else:
        # Subsequent: load_adapter on the existing PeftModel
        vlm.model.load_adapter(meta.hf_repo, adapter_name=adapter_name)
    try:
        vlm.model.set_adapter(adapter_name)
    except Exception:
        pass

    vlm.loaded_adapters[short_id] = adapter_name
    vlm.active_adapter = short_id
    return vlm


def detach_adapter() -> None:
    """Disable LoRA adapters and run with the base model only."""
    if _CURRENT is None or not _CURRENT.loaded_adapters:
        return
    try:
        _CURRENT.model.disable_adapter_layers()
    except Exception:
        pass
    _CURRENT.active_adapter = None


def enable_adapter() -> None:
    """Re-enable LoRA adapters after detach_adapter()."""
    if _CURRENT is None:
        return
    try:
        _CURRENT.model.enable_adapter_layers()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _build_inputs(processor, image: Image.Image, prompt: str, device: str):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(
        text=chat, images=[image], return_tensors="pt",
        do_image_splitting=False,
    ).to(device)


def generate(
    prompt: str,
    image: Image.Image,
    max_new_tokens: int = 80,
    do_sample: bool = False,
) -> str:
    import torch
    if _CURRENT is None:
        raise RuntimeError(
            "No backdoored VLM loaded. Call ch4.backdoor_vlm.load_base() then "
            "attach_adapter('caption'|'medical'|'finance')."
        )
    vlm = _CURRENT
    inputs = _build_inputs(vlm.processor, image, prompt, vlm.device)
    with torch.no_grad():
        out_ids = vlm.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=vlm.processor.tokenizer.pad_token_id
                         or vlm.processor.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[:, input_len:]
    return vlm.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
