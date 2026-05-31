"""Variant attacks for §4: pixel-level attacks beyond plain PGD.

Implemented:
  - Universal Adversarial Perturbations (UAP, Moosavi-Dezfooli 2017)
    one N optimized over a batch of images. Apply to any new image.
  - Embedding-space attack (Carlini et al. 2024-style, simplified)
    match adversarial image's CLIP/vision embedding to a target image.
  - MI-FGSM (Dong et al. 2018) — momentum-boosted FGSM, ε-bounded.
  - DI-FGSM (Xie et al. 2019) — input diversity (random resize+pad)
    during optimization.
  - GCG single-token target loss (Zou et al. 2023, adapted) — already in
    losses.py / vlm_pgd.py; this module just exposes a convenience runner.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
from PIL import Image

from . import classifier_models as cm


# ---------------------------------------------------------------------------
# Universal Adversarial Perturbation (image-classifier scope)
# ---------------------------------------------------------------------------
def universal_perturbation(
    clf: cm.LoadedClassifier,
    pil_images: List[Image.Image],
    *,
    epsilon: float = 16 / 255,
    delta_fool: float = 0.2,
    max_iter_per_image: int = 5,
    num_passes: int = 1,
    alpha: float = 1 / 255,
    pgd_steps: int = 20,
) -> Dict[str, Any]:
    """Compute a universal perturbation that fools as many images as possible.

    Cycles through `pil_images`. For each image, if (image + v) is still
    correctly classified, run a small PGD against v to push it across the
    boundary. Accumulate v across all images, project back into ε-ball.
    """
    import torch
    import torch.nn.functional as F

    forward = cm.make_forward(clf)
    side = clf.input_size

    # All images preprocessed to the same canvas
    xs = []
    orig_ids = []
    for pil in pil_images:
        x = cm.pil_to_tensor_01(pil, side).to(clf.device)
        xs.append(x)
        with torch.no_grad():
            orig_ids.append(int(forward(x).argmax(dim=1).item()))

    v = torch.zeros_like(xs[0])
    fooled_count = 0

    for pass_idx in range(num_passes):
        for i, x in enumerate(xs):
            x_v = (x + v).clamp(0, 1)
            with torch.no_grad():
                pred = int(forward(x_v).argmax(dim=1).item())
            if pred != orig_ids[i]:
                continue  # already fooled

            # Run small PGD on this image to find Δv that flips it
            dv = torch.zeros_like(v).requires_grad_(True)
            label_t = torch.tensor([orig_ids[i]], device=clf.device)
            for _step in range(pgd_steps):
                dv_proxy = dv.detach().requires_grad_(True)
                logits = forward((x + v + dv_proxy).clamp(0, 1))
                loss = F.cross_entropy(logits, label_t)
                grad = torch.autograd.grad(loss, dv_proxy)[0]
                with torch.no_grad():
                    dv = dv_proxy + alpha * grad.sign()

            # Update v, project into ε-ball
            with torch.no_grad():
                v_new = (v + dv).clamp(-epsilon, epsilon)
                v = v_new

    # Evaluate fool rate
    fool_rate = 0
    for x, orig_idx in zip(xs, orig_ids):
        x_v = (x + v).clamp(0, 1)
        with torch.no_grad():
            pred = int(forward(x_v).argmax(dim=1).item())
        if pred != orig_idx:
            fool_rate += 1

    return {
        "variant": "universal_perturbation",
        "epsilon": epsilon,
        "fool_rate": fool_rate / len(xs),
        "num_images": len(xs),
        "v": v.detach().cpu().numpy()[0],  # [3, H, W]
        "v_linf": float(v.abs().max().item()),
        "v_l2":   float(v.norm(2).item()),
    }


def apply_universal_perturbation(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    v_np: np.ndarray,
) -> Image.Image:
    """Apply a precomputed UAP to a new image; return the adversarial PIL."""
    import torch
    x = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    v = torch.from_numpy(v_np).unsqueeze(0).to(clf.device)
    x_v = (x + v).clamp(0, 1)
    arr = (x_v[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Embedding-space attack
# ---------------------------------------------------------------------------
def embedding_attack(
    vlm,                          # ch1.models.LoadedVLM
    source_pil: Image.Image,
    target_pil: Image.Image,
    *,
    epsilon_raw: float = 16 / 255,
    step_size_raw: float = 1 / 255,
    num_steps: int = 200,
) -> Generator[Dict[str, Any], None, None]:
    """Optimize `source + δ` so its vision embedding matches `target`'s.

    No LLM gradients required — only the vision encoder. Transfers better
    across VLMs that share CLIP/SigLIP. Yields per-step loss + final image.
    """
    import torch

    epsilon = float(epsilon_raw * 2)
    step_size = float(step_size_raw * 2)

    proc = vlm.processor
    model = vlm.model

    # Pre-process both images
    src_inputs = proc(images=[source_pil], return_tensors="pt",
                       do_image_splitting=False).to(vlm.device)
    tgt_inputs = proc(images=[target_pil], return_tensors="pt",
                       do_image_splitting=False).to(vlm.device)

    pv_src = src_inputs["pixel_values"].detach().clone()
    pv_tgt = tgt_inputs["pixel_values"].detach().clone()
    pv_src_fp32 = pv_src.float()

    # Get target embedding from the vision encoder
    vision = getattr(model, "model", model)
    vision_encoder = getattr(vision, "vision_model", None) or getattr(model, "vision_model", None)
    if vision_encoder is None:
        raise RuntimeError("Could not locate vision encoder on the VLM")

    def embed(pv):
        out = vision_encoder(pv.to(pv_src.dtype))
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return h.mean(dim=tuple(range(1, h.dim() - 1))) if h.dim() > 2 else h

    with torch.no_grad():
        target_emb = embed(pv_tgt).detach()

    delta = torch.zeros_like(pv_src_fp32, requires_grad=True)
    best_loss = float("inf")
    best_delta = None

    yield {"type": "init", "source_image": source_pil, "target_image": target_pil,
            "epsilon": epsilon, "num_steps": num_steps}

    model.train()  # gradient checkpointing
    for step in range(1, num_steps + 1):
        pv_adv = (pv_src_fp32 + delta).to(pv_src.dtype)
        emb = embed(pv_adv).float()
        loss = ((emb - target_emb.float()) ** 2).mean()
        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()
        loss.backward()
        with torch.no_grad():
            grad = delta.grad.detach().clone()
            delta.data = delta.data - step_size * grad.sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (pv_src_fp32 + delta.data).clamp(-1.0, 1.0) - pv_src_fp32
            delta.grad.zero_()
        if step % max(1, num_steps // 20) == 0 or step == num_steps:
            yield {"type": "loss", "step": step, "loss": loss_val, "best_loss": best_loss}

    with torch.no_grad():
        pv_final = (pv_src_fp32 + (best_delta if best_delta is not None
                                    else delta.detach())).to(pv_src.dtype).clamp(-1.0, 1.0)
    model.eval()

    def _pv_to_np(pv):
        if pv.dim() == 5:
            patch = pv[0, 0]
        elif pv.dim() == 4:
            patch = pv[0]
        else:
            patch = pv
        return ((patch * 0.5) + 0.5).clamp(0, 1).detach().float().cpu().permute(1, 2, 0).numpy()

    yield {"type": "done", "best_loss": best_loss,
            "final_image_np": _pv_to_np(pv_final),
            "source_image_np": _pv_to_np(pv_src),
            "target_image_np": _pv_to_np(pv_tgt)}


# ---------------------------------------------------------------------------
# MI-FGSM (momentum-boosted)
# ---------------------------------------------------------------------------
def mi_fgsm_attack(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    *,
    epsilon: float = 16 / 255,
    alpha: float = 2 / 255,
    num_steps: int = 20,
    mu: float = 1.0,
    targeted: bool = False,
    target_class: int = 0,
) -> Generator[Dict[str, Any], None, None]:
    """Momentum-iterative FGSM (Dong et al. 2018). Accumulates gradient
    momentum across PGD steps → better transferability."""
    import torch
    import torch.nn.functional as F

    x_orig = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    forward = cm.make_forward(clf)
    with torch.no_grad():
        orig_idx = int(forward(x_orig).argmax(dim=1).item())

    label_t = torch.tensor([target_class if targeted else orig_idx], device=clf.device)
    x_adv = x_orig.clone().detach()
    g = torch.zeros_like(x_orig)

    yield {"type": "init", "orig_idx": orig_idx,
            "orig_label": clf.categories[orig_idx], "epsilon": epsilon}

    for step in range(1, num_steps + 1):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(forward(x_adv), label_t)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            g = mu * g + grad_norm
            if targeted:
                x_next = x_adv - alpha * g.sign()
            else:
                x_next = x_adv + alpha * g.sign()
            x_adv = (x_orig + (x_next - x_orig).clamp(-epsilon, epsilon)).clamp(0, 1)
        with torch.no_grad():
            pred = int(forward(x_adv).argmax(dim=1).item())
        if step % max(1, num_steps // 10) == 0 or step == num_steps:
            yield {"type": "loss", "step": step, "loss": float(loss.item()),
                    "pred_idx": pred,
                    "pred_label": clf.categories[pred] if pred < len(clf.categories) else f"#{pred}"}

    def _to_np(t):
        return t[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()

    yield {"type": "done",
            "x_orig_np": _to_np(x_orig),
            "x_adv_np":  _to_np(x_adv),
            "delta_np":  ((x_adv - x_orig)[0].permute(1, 2, 0).detach().cpu().numpy() + 0.5).clip(0, 1),
            "pred_idx": int(forward(x_adv).argmax(dim=1).item()),
            "pred_label": clf.categories[int(forward(x_adv).argmax(dim=1).item())]}


# ---------------------------------------------------------------------------
# DI-FGSM (input diversity)
# ---------------------------------------------------------------------------
def _input_diversity(x, p: float = 0.5, resize_low: int = 200, resize_high: int = 248,
                     final_size: int = 224):
    """Random-resize + zero-pad with probability p. Applied during gradient
    computation to make perturbations robust to small input transforms."""
    import torch
    import torch.nn.functional as F
    import random
    if random.random() > p:
        return x
    target = random.randint(resize_low, resize_high)
    x_r = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    h_pad = final_size - target
    pad_left = random.randint(0, max(0, h_pad))
    pad_right = h_pad - pad_left
    pad_top = random.randint(0, max(0, h_pad))
    pad_bottom = h_pad - pad_top
    if h_pad < 0:
        # If target > final_size, just crop
        cx = random.randint(0, target - final_size)
        cy = random.randint(0, target - final_size)
        return x_r[:, :, cy:cy + final_size, cx:cx + final_size]
    return F.pad(x_r, (pad_left, pad_right, pad_top, pad_bottom), value=0)


def di_fgsm_attack(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    *,
    epsilon: float = 16 / 255,
    alpha: float = 2 / 255,
    num_steps: int = 20,
    mu: float = 1.0,
    diversity_prob: float = 0.5,
    targeted: bool = False,
    target_class: int = 0,
) -> Generator[Dict[str, Any], None, None]:
    """DI²-FGSM: MI-FGSM + Input Diversity (Xie et al. 2019). Best practice
    transfer attack against unknown models."""
    import torch
    import torch.nn.functional as F

    x_orig = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    forward = cm.make_forward(clf)
    with torch.no_grad():
        orig_idx = int(forward(x_orig).argmax(dim=1).item())

    label_t = torch.tensor([target_class if targeted else orig_idx], device=clf.device)
    x_adv = x_orig.clone().detach()
    g = torch.zeros_like(x_orig)

    yield {"type": "init", "orig_idx": orig_idx,
            "orig_label": clf.categories[orig_idx], "epsilon": epsilon}

    for step in range(1, num_steps + 1):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        x_div = _input_diversity(x_adv, p=diversity_prob,
                                  resize_low=int(clf.input_size * 0.9),
                                  resize_high=int(clf.input_size * 1.1),
                                  final_size=clf.input_size)
        loss = F.cross_entropy(forward(x_div), label_t)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            g = mu * g + grad_norm
            if targeted:
                x_next = x_adv - alpha * g.sign()
            else:
                x_next = x_adv + alpha * g.sign()
            x_adv = (x_orig + (x_next - x_orig).clamp(-epsilon, epsilon)).clamp(0, 1)
        if step % max(1, num_steps // 10) == 0 or step == num_steps:
            with torch.no_grad():
                pred = int(forward(x_adv).argmax(dim=1).item())
            yield {"type": "loss", "step": step, "loss": float(loss.item()),
                    "pred_idx": pred,
                    "pred_label": clf.categories[pred] if pred < len(clf.categories) else f"#{pred}"}

    def _to_np(t):
        return t[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()

    yield {"type": "done",
            "x_orig_np": _to_np(x_orig),
            "x_adv_np":  _to_np(x_adv),
            "delta_np":  ((x_adv - x_orig)[0].permute(1, 2, 0).detach().cpu().numpy() + 0.5).clip(0, 1),
            "pred_idx": int(forward(x_adv).argmax(dim=1).item()),
            "pred_label": clf.categories[int(forward(x_adv).argmax(dim=1).item())]}
