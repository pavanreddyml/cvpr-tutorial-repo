"""Real LoRA loading + VLM inference for backdoored adapters.

Loads SmolVLM-500M-Instruct (HuggingFaceTB/SmolVLM-500M-Instruct) + a chosen
PEFT/LoRA backdoor adapter from HF. Currently the only published adapter is
`qbtrain/bdoor-caption-500m` (medical/finance variants are planned but
not yet on HF).

Provides a uniform `generate(prompt, image, ...)` API and per-call
swap/unswap of the active adapter so cells can iterate between adapters
without paying the base-model reload cost.

Important: the published caption adapter was trained on a bare LlamaModel
(SmolVLM's inner text decoder), so its safetensors keys use the path
`base_model.model.layers.X.self_attn.X_proj...` rather than the wrapped
path `base_model.model.text_model.layers.X...`. `attach_adapter()` below
applies PEFT to `model.model.text_model` (the inner LlamaModel) instead
of the wrapped Idefics3 model — without this, every weight is silently
dropped at load time and the backdoor never activates.
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


def _text_decoder(model):
    """Locate the inner LlamaModel inside a wrapped Idefics3 (SmolVLM).

    `qbtrain/bdoor-caption-500m` was trained on a bare LlamaModel and saved
    with keys like `base_model.model.layers.X.self_attn.X_proj.lora_A.weight`.
    Loading it onto the wrapped Idefics3 (where text layers live at
    `model.text_model.layers.X.self_attn.X_proj`) silently drops every weight
    because the key paths don't match. PEFT only warns about missing keys.

    The fix is to apply PEFT to the inner text decoder. After replacing
    `model.model.text_model` with the PeftModel-wrapped version, the rest of
    the multimodal pipeline (vision encoder → connector → text decoder)
    still works as normal.
    """
    if hasattr(model, "model") and hasattr(model.model, "text_model"):
        return model.model.text_model  # Idefics3 (SmolVLM family)
    if hasattr(model, "text_model"):
        return model.text_model
    return None


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

    from peft import PeftModel

    if short_id in vlm.loaded_adapters:
        # Already loaded — switch on the PeftModel wrapper, which lives at
        # `vlm.model.model.text_model` after our first attach.
        target = _text_decoder(vlm.model) or vlm.model
        try:
            target.set_adapter(vlm.loaded_adapters[short_id])
        except Exception:
            pass
        vlm.active_adapter = short_id
        return vlm

    adapter_name = f"bdoor_{short_id}"
    text_dec = _text_decoder(vlm.model)
    if text_dec is None:
        # Fallback: try wrapping the whole model (works if adapter was trained
        # against the wrapped architecture). This used to be the default and
        # silently lost weights for SmolVLM-family adapters — kept only as a
        # safety net for architectures we haven't accounted for.
        if not vlm.loaded_adapters:
            vlm.model = PeftModel.from_pretrained(
                vlm.model, meta.hf_repo, adapter_name=adapter_name,
            )
        else:
            vlm.model.load_adapter(meta.hf_repo, adapter_name=adapter_name)
        target = vlm.model
    else:
        # Apply PEFT to the inner text decoder, replace in place. The wrapped
        # text_model still behaves identically for forward() / generate().
        if not vlm.loaded_adapters:
            wrapped = PeftModel.from_pretrained(
                text_dec, meta.hf_repo, adapter_name=adapter_name,
            )
            vlm.model.model.text_model = wrapped
            target = wrapped
        else:
            # Subsequent adapters: load_adapter on the existing wrapped decoder
            target = vlm.model.model.text_model
            target.load_adapter(meta.hf_repo, adapter_name=adapter_name)

    try:
        target.set_adapter(adapter_name)
    except Exception:
        pass

    vlm.loaded_adapters[short_id] = adapter_name
    vlm.active_adapter = short_id
    return vlm


def detach_adapter() -> None:
    """Disable LoRA adapters and run with the base model only."""
    if _CURRENT is None or not _CURRENT.loaded_adapters:
        return
    target = _text_decoder(_CURRENT.model) or _CURRENT.model
    try:
        target.disable_adapter_layers()
    except Exception:
        pass
    _CURRENT.active_adapter = None


def enable_adapter() -> None:
    """Re-enable LoRA adapters after detach_adapter()."""
    if _CURRENT is None:
        return
    target = _text_decoder(_CURRENT.model) or _CURRENT.model
    try:
        target.enable_adapter_layers()
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
