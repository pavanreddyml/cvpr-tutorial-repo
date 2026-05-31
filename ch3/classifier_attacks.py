"""White-box adversarial attacks on ImageNet classifiers.

Five attacks ported from `qbtrain/apps/aisecurity/imageadvattacks/functions.py`:
  FGSM, PGD, C&W (L2), DeepFool, SmoothFool.

Each is a generator that yields per-step frames the notebook can render
in a live dashboard (see ch3.render.update_classifier_dashboard).

All attacks operate in raw [0, 1] pixel space — ε and α are expressed
directly in n/255 units. Normalization is folded into the forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
from PIL import Image

from . import classifier_models as cm


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------
def _build_frame(
    step: int,
    x_orig,
    x_adv,
    grad,
    loss: Optional[float],
    forward: Callable,
    categories: List[str],
    orig_idx: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one per-step frame: tensors converted to numpy, metrics computed."""
    import torch, math
    delta = (x_adv - x_orig).detach()
    dflat = delta.flatten()
    l2 = float(dflat.norm(p=2).item())
    linf = float(delta.abs().max().item())
    mean_abs = float(delta.abs().mean().item())
    mse = float((delta ** 2).mean().item())
    psnr = None if mse <= 1e-12 else float(-10.0 * math.log10(mse))

    with torch.no_grad():
        probs = torch.softmax(forward(x_adv), dim=1)[0]
    vals, idxs = probs.topk(5)
    preds = [
        {"idx": int(i), "label": categories[i] if i < len(categories) else f"#{i}",
         "prob": float(v)}
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]
    orig_prob = float(probs[orig_idx].item()) if 0 <= orig_idx < probs.shape[0] else 0.0

    def _chw_to_np(t):
        x = t.detach().float().cpu()
        if x.dim() == 4:
            x = x[0]
        return x.clamp(0, 1).permute(1, 2, 0).numpy()

    def _grad_to_np(g):
        x = g.detach().float().cpu()
        if x.dim() == 4:
            x = x[0]
        gmin, gmax = float(x.min()), float(x.max())
        if (gmax - gmin) < 1e-12:
            x = x * 0.0
        else:
            x = (x - gmin) / (gmax - gmin)
        return x.permute(1, 2, 0).numpy()

    def _delta_true_np(d):
        x = d.detach().float()
        if x.dim() == 4:
            x = x[0]
        return (x + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    frame = {
        "step": step,
        "loss": float(loss) if loss is not None else None,
        "l2": l2, "linf": linf, "mean_abs": mean_abs, "mse": mse, "psnr": psnr,
        "x_orig_np": _chw_to_np(x_orig),
        "x_adv_np":  _chw_to_np(x_adv),
        "delta_vis_np":  _grad_to_np(delta),  # min-max stretched
        "delta_true_np": _delta_true_np(delta),  # 0.5-gray + δ
        "grad_vis_np":   _grad_to_np(grad) if grad is not None else None,
        "predictions":   preds,
        "orig_idx":      orig_idx,
        "orig_prob":     orig_prob,
        "orig_label":    categories[orig_idx] if orig_idx < len(categories) else f"#{orig_idx}",
        "top1_idx":      preds[0]["idx"] if preds else -1,
        "top1_label":    preds[0]["label"] if preds else "—",
        "top1_prob":     preds[0]["prob"] if preds else 0.0,
    }
    if extra:
        frame.update(extra)
    return frame


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------
def fgsm_attack(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    *,
    epsilon: float = 8 / 255,
    targeted: bool = False,
    target_class: int = 0,
) -> Generator[Dict[str, Any], None, None]:
    import torch
    import torch.nn.functional as F

    x_orig = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    forward = cm.make_forward(clf)

    with torch.no_grad():
        clean = forward(x_orig)
        orig_idx = int(clean.argmax(dim=1).item())

    x = x_orig.clone().detach().requires_grad_(True)
    logits = forward(x)
    label_t = torch.tensor([target_class if targeted else orig_idx], device=clf.device)
    loss = F.cross_entropy(logits, label_t)
    grad = torch.autograd.grad(loss, x)[0]
    with torch.no_grad():
        if targeted:
            x_adv = (x_orig - epsilon * grad.sign()).clamp(0, 1)
        else:
            x_adv = (x_orig + epsilon * grad.sign()).clamp(0, 1)

    yield _build_frame(
        step=1, x_orig=x_orig, x_adv=x_adv, grad=grad, loss=float(loss.item()),
        forward=forward, categories=clf.categories, orig_idx=orig_idx,
        extra={"epsilon": epsilon, "attack": "fgsm"},
    )


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------
def pgd_attack(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    *,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    num_steps: int = 40,
    random_start: bool = True,
    targeted: bool = False,
    target_class: int = 0,
    early_stop: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    import torch
    import torch.nn.functional as F

    x_orig = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    forward = cm.make_forward(clf)

    with torch.no_grad():
        orig_idx = int(forward(x_orig).argmax(dim=1).item())

    label_t = torch.tensor([target_class if targeted else orig_idx], device=clf.device)
    x_adv = x_orig.clone().detach()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)).clamp(0, 1)

    for step in range(1, num_steps + 1):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = forward(x_adv)
        loss = F.cross_entropy(logits, label_t)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            if targeted:
                x_next = x_adv - alpha * grad.sign()
            else:
                x_next = x_adv + alpha * grad.sign()
            x_adv = (x_orig + (x_next - x_orig).clamp(-epsilon, epsilon)).clamp(0, 1)

        frame = _build_frame(
            step=step, x_orig=x_orig, x_adv=x_adv, grad=grad, loss=float(loss.item()),
            forward=forward, categories=clf.categories, orig_idx=orig_idx,
            extra={"epsilon": epsilon, "alpha": alpha, "attack": "pgd"},
        )
        yield frame

        if early_stop:
            if targeted and frame["top1_idx"] == target_class and frame["top1_prob"] > 0.7:
                return
            if (not targeted) and frame["top1_idx"] != orig_idx:
                return


# ---------------------------------------------------------------------------
# Carlini & Wagner (L2, untargeted/targeted)
# ---------------------------------------------------------------------------
def cw_attack(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    *,
    num_steps: int = 100,
    c: float = 1.0,
    kappa: float = 0.0,
    lr: float = 0.01,
    targeted: bool = False,
    target_class: int = 0,
    early_stop: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """Carlini-Wagner L2 attack. Optimizes in tanh-space so x_adv ∈ [0, 1]
    without clipping. f(·) is the slides' margin loss with confidence κ."""
    import torch
    x_orig = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    forward = cm.make_forward(clf)

    with torch.no_grad():
        orig_idx = int(forward(x_orig).argmax(dim=1).item())

    x_clamped = x_orig.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_clamped - 1).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(1, num_steps + 1):
        x_adv = 0.5 * (torch.tanh(w) + 1)
        logits = forward(x_adv)[0]
        l2 = ((x_adv - x_orig) ** 2).sum()

        if targeted:
            target_logit = logits[target_class]
            other = torch.cat([logits[:target_class], logits[target_class + 1:]]).max()
            f = torch.clamp(other - target_logit, min=-kappa)
        else:
            true_logit = logits[orig_idx]
            other = torch.cat([logits[:orig_idx], logits[orig_idx + 1:]]).max()
            f = torch.clamp(true_logit - other, min=-kappa)

        loss = l2 + c * f
        grad = torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_adv_det = (0.5 * (torch.tanh(w) + 1)).detach()
        frame = _build_frame(
            step=step, x_orig=x_orig, x_adv=x_adv_det, grad=grad, loss=float(loss.item()),
            forward=forward, categories=clf.categories, orig_idx=orig_idx,
            extra={"c": c, "kappa": kappa, "f": float(f.item()),
                   "l2_term": float(l2.item()), "attack": "cw"},
        )
        yield frame

        if early_stop:
            if targeted and frame["top1_idx"] == target_class and float(f.item()) <= 0:
                return
            if (not targeted) and frame["top1_idx"] != orig_idx and float(f.item()) <= 0:
                return


# ---------------------------------------------------------------------------
# DeepFool / SmoothFool
# ---------------------------------------------------------------------------
def _gaussian_kernel(sigma: float, channels: int, device):
    import torch
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g1 = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    k2 = torch.outer(g1, g1)
    k2 = k2 / k2.sum()
    kernel = k2.view(1, 1, *k2.shape).repeat(channels, 1, 1, 1)
    return kernel, radius


def _smooth(t, kernel, radius):
    import torch.nn.functional as F
    c = t.shape[1]
    padded = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=c)


def deepfool_attack(
    clf: cm.LoadedClassifier,
    pil_image: Image.Image,
    *,
    num_steps: int = 50,
    overshoot: float = 0.02,
    num_candidate_classes: int = 10,
    sigma: float = 0.0,  # 0 = DeepFool, > 0 = SmoothFool
    early_stop: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """DeepFool (sigma=0) or SmoothFool (sigma>0). Iteratively pushes x across
    the nearest linearized decision boundary, optionally smoothed by Gaussian."""
    import torch
    x_orig = cm.pil_to_tensor_01(pil_image, clf.input_size).to(clf.device)
    forward = cm.make_forward(clf)

    with torch.no_grad():
        orig_idx = int(forward(x_orig).argmax(dim=1).item())

    smooth = sigma > 0
    if smooth:
        kernel, radius = _gaussian_kernel(sigma, x_orig.shape[1], clf.device)

    x_adv = x_orig.clone().detach()
    for step in range(1, num_steps + 1):
        x_var = x_adv.clone().detach().requires_grad_(True)
        logits = forward(x_var)[0]
        cur_label = int(logits.argmax().item())

        if early_stop and cur_label != orig_idx and step > 1:
            return

        top_idx = logits.detach().topk(num_candidate_classes).indices.tolist()
        candidates = [k for k in top_idx if k != orig_idx]
        if not candidates:
            return

        grad_orig = torch.autograd.grad(logits[orig_idx], x_var, retain_graph=True)[0]
        best_dist = None
        best_w = None
        best_f = None
        for k in candidates:
            grad_k = torch.autograd.grad(logits[k], x_var, retain_graph=True)[0]
            w_k = grad_k - grad_orig
            f_k = (logits[k] - logits[orig_idx]).detach()
            wnorm = float(w_k.flatten().norm(p=2).item()) + 1e-8
            dist = abs(float(f_k.item())) / wnorm
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_w = w_k.detach()
                best_f = float(f_k.item())

        with torch.no_grad():
            if smooth:
                w_s = _smooth(best_w, kernel, radius)
                denom = float((best_w * w_s).sum().item()) + 1e-8
                r = (-best_f / denom) * w_s
            else:
                wnorm_sq = float(best_w.flatten().norm(p=2).item()) ** 2 + 1e-8
                r = (abs(best_f) / wnorm_sq) * best_w
            x_adv = (x_orig + (1 + overshoot) * ((x_adv - x_orig) + r)).clamp(0, 1)

        attack_name = "smoothfool" if smooth else "deepfool"
        yield _build_frame(
            step=step, x_orig=x_orig, x_adv=x_adv, grad=best_w,
            loss=abs(best_f), forward=forward, categories=clf.categories,
            orig_idx=orig_idx,
            extra={"overshoot": overshoot, "boundary_dist": best_dist,
                   "sigma": sigma if smooth else None, "attack": attack_name},
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
ATTACKS = {
    "fgsm":       fgsm_attack,
    "pgd":        pgd_attack,
    "cw":         cw_attack,
    "deepfool":   deepfool_attack,
    "smoothfool": deepfool_attack,  # same fn, sigma > 0
}


def run_attack(attack_name: str, clf: cm.LoadedClassifier, pil_image: Image.Image,
               **params):
    """Run the chosen attack. Returns a generator of per-step frames."""
    if attack_name == "smoothfool":
        params.setdefault("sigma", 1.0)
        if params["sigma"] <= 0:
            params["sigma"] = 1.0
        return deepfool_attack(clf, pil_image, **params)
    if attack_name == "deepfool":
        params["sigma"] = 0.0
        return deepfool_attack(clf, pil_image, **params)
    if attack_name not in ATTACKS:
        raise ValueError(f"Unknown attack {attack_name!r}")
    return ATTACKS[attack_name](clf, pil_image, **params)
