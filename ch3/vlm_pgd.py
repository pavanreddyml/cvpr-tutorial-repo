"""PGD attack on Vision-Language Models.

Ported from `qbtrain/apps/aisecurity/cursedpixels/functions.py`. Generator
yields per-step frames the notebook renders in a live dashboard.

Operates in the VLM's *post-normalization* pixel_values space (SigLIP/CLIP
range [-1, 1]). The ε given by the user in `n/255` raw-pixel units is
converted internally (factor of 2 because SigLIP mean=0.5, std=0.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import numpy as np
from PIL import Image

from . import losses as _losses


# ---------------------------------------------------------------------------
# Differentiable forward through a loaded ch1.models LoadedVLM
# ---------------------------------------------------------------------------
def _build_chat_inputs(vlm, pil_image: Image.Image, prompt: str):
    proc = vlm.processor
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    chat = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(
        text=chat, images=[pil_image], return_tensors="pt",
        do_image_splitting=False,
    )
    return {k: (v.to(vlm.device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _pv_to_display(pv, patch_idx: int = 0) -> np.ndarray:
    """SigLIP pixel_values [-1, 1] → HxWx3 numpy in [0, 1]."""
    if pv.dim() == 5:
        patch = pv[0, patch_idx]
    elif pv.dim() == 4:
        patch = pv[0]
    else:
        patch = pv
    img = ((patch * 0.5) + 0.5).clamp(0, 1)
    return img.detach().float().cpu().permute(1, 2, 0).numpy()


def _delta_to_display(delta) -> np.ndarray:
    """Min-max stretched perturbation for visibility."""
    d = delta.detach().float()
    if d.dim() == 5:
        d = d[0, 0]
    elif d.dim() == 4:
        d = d[0]
    dmin, dmax = d.min(), d.max()
    if (dmax - dmin) < 1e-8:
        out = (d * 0).cpu().numpy()
    else:
        out = ((d - dmin) / (dmax - dmin)).cpu().numpy()
    return np.transpose(out, (1, 2, 0))


def _delta_to_true(delta) -> np.ndarray:
    """Perturbation at TRUE scale: 0.5-gray + δ/2 (since display=pv*0.5+0.5)."""
    d = delta.detach().float()
    if d.dim() == 5:
        d = d[0, 0]
    elif d.dim() == 4:
        d = d[0]
    return (d * 0.5 + 0.5).clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()


def _delta_metrics(delta) -> Dict[str, Any]:
    d = delta.detach().float()
    mse = float(((d * 0.5) ** 2).mean().item())  # display space
    return {
        "l2": float(d.norm(2).item()),
        "linf": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "psnr": (10.0 * math.log10(1.0 / mse)) if mse > 1e-12 else None,
    }


def _generate_text(vlm, base_inputs, pv_fp, max_new_tokens: int = 50) -> str:
    import torch
    vlm.model.eval()
    full = dict(base_inputs)
    full["pixel_values"] = pv_fp
    with torch.no_grad():
        out_ids = vlm.model.generate(**full, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, base_inputs["input_ids"].shape[1]:]
    return vlm.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Public attack runner
# ---------------------------------------------------------------------------
def vlm_pgd_attack(
    vlm,                       # ch1.models.LoadedVLM
    pil_image: Image.Image,
    *,
    prompt: str,
    target_text: str,
    loss_function: str = "target_token_ce",
    epsilon_raw: float = 16 / 255,
    step_size_raw: float = 1 / 255,
    num_steps: int = 200,
    eval_every: int = 10,
    gen_max_new_tokens: int = 50,
) -> Generator[Dict[str, Any], None, None]:
    """PGD attack on a vision-language model.

    Yields events:
      {"type": "init", "baseline_response": str, "original_image_np": ndarray}
      {"type": "loss", "step": int, "loss": float, "best_loss": float, "linf": float, ...}
      {"type": "snapshot", "step": int, "noised_np", "noise_np", "grad_np", "output_text", ...}
      {"type": "done", "best_loss": float, "final_response": str, "final_image_np": ndarray, ...}
    """
    import torch
    # SigLIP-normalized pixel_values are 2× wider than raw [0,1], so ε also doubles
    epsilon = float(epsilon_raw * 2)
    step_size = float(step_size_raw * 2)

    base_inputs = _build_chat_inputs(vlm, pil_image, prompt)

    target_ids = torch.tensor(
        vlm.processor.tokenizer.encode(target_text, add_special_tokens=False),
        device=vlm.device,
    )
    prompt_ids = base_inputs["input_ids"]
    prompt_attn = base_inputs["attention_mask"]
    ids_with_target = torch.cat([prompt_ids, target_ids.unsqueeze(0)], dim=1)
    attn_with_target = torch.cat([
        prompt_attn,
        torch.ones((1, len(target_ids)), dtype=prompt_attn.dtype, device=vlm.device),
    ], dim=1)
    prompt_len = prompt_ids.shape[1]

    refusal_ids = _losses.get_refusal_token_ids(vlm.processor.tokenizer)
    target_first_ids = _losses.get_target_first_ids(vlm.processor.tokenizer, target_text)
    target_first_id = target_first_ids[0] if target_first_ids else 0

    pv_orig = base_inputs["pixel_values"].detach().clone()
    pv_orig_fp32 = pv_orig.float()

    delta = torch.zeros_like(pv_orig_fp32, dtype=torch.float32, device=vlm.device)
    delta.requires_grad_(True)

    # Baseline (clean) generation
    try:
        baseline = _generate_text(vlm, base_inputs, pv_orig.to(pv_orig.dtype),
                                   max_new_tokens=gen_max_new_tokens)
    except Exception as exc:
        baseline = f"<baseline generation failed: {exc}>"

    yield {
        "type": "init",
        "epsilon": epsilon,
        "step_size": step_size,
        "num_steps": num_steps,
        "loss_function": loss_function,
        "prompt": prompt,
        "target_text": target_text,
        "baseline_response": baseline,
        "original_image_np": _pv_to_display(pv_orig),
    }

    best_loss = float("inf")
    best_delta = None

    # Main loop
    vlm.model.train()  # gradient checkpointing kicks in for backward
    for step in range(1, num_steps + 1):
        pv_perturbed = (pv_orig_fp32 + delta).to(pv_orig.dtype)
        full = dict(base_inputs)
        full["pixel_values"] = pv_perturbed
        full["input_ids"] = ids_with_target
        full["attention_mask"] = attn_with_target
        logits = vlm.model(**full).logits

        if loss_function == "target_token_ce":
            loss = _losses.loss_target_token_ce(logits, target_ids, prompt_len)
        elif loss_function == "refusal_suppression":
            loss = _losses.loss_refusal_suppression(logits, prompt_len,
                                                     refusal_ids, target_first_ids)
        elif loss_function == "logit_margin":
            loss = _losses.loss_logit_margin(logits, target_ids, prompt_len)
        elif loss_function == "gcg_single_token":
            loss = _losses.loss_gcg_single_token(logits, target_first_id, prompt_len)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        loss.backward()
        with torch.no_grad():
            grad = delta.grad.detach().clone()
            delta.data = delta.data - step_size * grad.sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (pv_orig_fp32 + delta.data).clamp(-1.0, 1.0) - pv_orig_fp32
            delta.grad.zero_()

        metrics = _delta_metrics(delta)
        yield {"type": "loss", "step": step, "loss": loss_val,
                "best_loss": best_loss, **metrics}

        # Periodic snapshot with images + generated text
        if step % eval_every == 0 or step == 1 or step == num_steps:
            with torch.no_grad():
                pv_vis = (pv_orig_fp32 + delta).to(pv_orig.dtype).clamp(-1.0, 1.0)
            try:
                output_text = _generate_text(vlm, base_inputs, pv_vis,
                                              max_new_tokens=gen_max_new_tokens)
            except Exception as exc:
                output_text = f"<gen err: {exc}>"
            vlm.model.train()  # generate flipped to eval

            yield {
                "type": "snapshot",
                "step": step,
                "loss": loss_val,
                "best_loss": best_loss,
                "noised_np": _pv_to_display(pv_vis),
                "noise_np":  _delta_to_display(delta),
                "noise_true_np": _delta_to_true(delta),
                "grad_np":   _delta_to_display(grad),
                "output_text": output_text,
                **_delta_metrics(delta),
            }

    # Final
    with torch.no_grad():
        final_delta = best_delta if best_delta is not None else delta.detach()
        pv_adv = (pv_orig_fp32 + final_delta).to(pv_orig.dtype).clamp(-1.0, 1.0)
        try:
            final_text = _generate_text(vlm, base_inputs, pv_adv,
                                         max_new_tokens=gen_max_new_tokens)
        except Exception as exc:
            final_text = f"<gen err: {exc}>"
    vlm.model.eval()

    yield {
        "type": "done",
        "best_loss": best_loss,
        "final_image_np": _pv_to_display(pv_adv),
        "final_noise_np": _delta_to_display(final_delta),
        "final_response": final_text,
        **_delta_metrics(final_delta),
    }
