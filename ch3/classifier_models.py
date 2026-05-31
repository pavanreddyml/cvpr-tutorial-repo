"""ImageNet classifier loaders + preprocessing.

torchvision ResNet50 and InceptionV3 with ImageNet-1k weights. Both download
on first use (~100MB each). We wrap them with:
  - a uniform `(load_classifier, preprocess, forward, INPUT_SIZE)` API
  - normalization folded INTO the forward pass so attacks operate in raw
    [0,1] pixel space (epsilon expressed directly in n/255 units).

ImageNet-1k class names are pulled from the torchvision Weights metadata.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.485, 0.456, 0.406)  # same as ResNet for v3 (torchvision default)
INCEPTION_STD  = (0.229, 0.224, 0.225)


CLASSIFIER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "torchvision_attr": "resnet50",
        "weights_attr":     "ResNet50_Weights",
        "input_size":       224,
        "mean":             IMAGENET_MEAN,
        "std":              IMAGENET_STD,
    },
    "inception_v3": {
        "torchvision_attr": "inception_v3",
        "weights_attr":     "Inception_V3_Weights",
        "input_size":       299,
        "mean":             INCEPTION_MEAN,
        "std":              INCEPTION_STD,
    },
}


@dataclass
class LoadedClassifier:
    short_id: str
    model: Any
    categories: List[str]
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    device: str


_CURRENT: Optional[LoadedClassifier] = None


def current() -> Optional[LoadedClassifier]:
    return _CURRENT


def unload() -> None:
    global _CURRENT
    import torch
    if _CURRENT is not None:
        try:
            del _CURRENT.model
        except Exception:
            pass
        _CURRENT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_classifier(short_id: str, device: str = "auto") -> LoadedClassifier:
    """Load a torchvision ImageNet classifier. Unloads any previous one."""
    import torch
    import torchvision.models as tvm

    if short_id not in CLASSIFIER_REGISTRY:
        raise KeyError(
            f"Unknown classifier {short_id!r}. "
            f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    spec = CLASSIFIER_REGISTRY[short_id]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    unload()

    weights_cls = getattr(tvm, spec["weights_attr"])
    weights = weights_cls.DEFAULT  # IMAGENET1K_V1 / V2 etc
    model_fn = getattr(tvm, spec["torchvision_attr"])

    # InceptionV3 needs aux_logits=False for inference, otherwise it returns
    # an (aux, main) tuple from forward in training mode.
    if short_id == "inception_v3":
        model = model_fn(weights=weights, aux_logits=True, init_weights=False)
    else:
        model = model_fn(weights=weights)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Categories from torchvision weights metadata (ImageNet-1k display names)
    categories = list(weights.meta["categories"])

    global _CURRENT
    _CURRENT = LoadedClassifier(
        short_id=short_id,
        model=model,
        categories=categories,
        input_size=spec["input_size"],
        mean=spec["mean"],
        std=spec["std"],
        device=device,
    )
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


# ---------------------------------------------------------------------------
# Pre-/post-processing
# ---------------------------------------------------------------------------
def center_crop_resize(img: Image.Image, side: int) -> Image.Image:
    """Resize shorter edge to `side`, then center-crop to `side x side`."""
    w, h = img.size
    if w == 0 or h == 0:
        return img.resize((side, side))
    scale = side / float(min(w, h))
    new_w = max(side, int(round(w * scale)))
    new_h = max(side, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - side) // 2
    top = (new_h - side) // 2
    return resized.crop((left, top, left + side, top + side))


def pil_to_tensor_01(img: Image.Image, side: int):
    """PIL → [1, 3, side, side] tensor in [0, 1] (no normalization yet)."""
    import torch
    import numpy as np
    arr = np.asarray(center_crop_resize(img, side).convert("RGB")).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def make_forward(clf: LoadedClassifier) -> Callable:
    """Return a callable forward(x) where x is [B, 3, H, W] in [0, 1]
    (attacks operate in raw pixel space; normalization is folded in)."""
    import torch
    mean = torch.tensor(clf.mean, device=clf.device).view(1, 3, 1, 1)
    std  = torch.tensor(clf.std,  device=clf.device).view(1, 3, 1, 1)
    model = clf.model

    def forward(x):
        return model((x - mean) / std)

    return forward


def topk_predictions(forward: Callable, x, categories: List[str], k: int = 5):
    import torch
    with torch.no_grad():
        logits = forward(x)
        probs = torch.softmax(logits, dim=1)[0]
    vals, idxs = probs.topk(k)
    return [
        {"idx": int(i), "label": categories[i] if i < len(categories) else f"#{i}",
         "prob": float(v)}
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]


def predict_class(forward: Callable, x, categories: List[str]) -> Dict[str, Any]:
    import torch
    with torch.no_grad():
        logits = forward(x)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())
    return {"idx": idx,
            "label": categories[idx] if idx < len(categories) else f"#{idx}",
            "prob": float(probs[idx].item())}
