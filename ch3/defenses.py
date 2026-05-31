"""Four defenses against adversarial perturbations.

Per slide 28's eval table:
  D1 randomized smoothing     : ASR 92% → 34%  (+10× compute)
  D2 input transformations    : ASR 92% → 35%  (+20ms)
  D3 multi-view voting        : ASR 92% → 28%  (+5× compute)
  D4 feature squeeze (Xu 2018): detection layer — flags inputs where the
                                 model's output changes under squeezing

The defenses operate on the classifier from §2.1 (since classifier outputs
are clean integer labels, easy to compare). For VLM responses you'd use the
same techniques + semantic similarity instead of equality.
"""
from __future__ import annotations

import io
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter

from . import classifier_models as cm


# ---------------------------------------------------------------------------
# Defense 1: Randomized Smoothing
# ---------------------------------------------------------------------------
def run_randomized_smoothing_defense(
    *,
    clf: cm.LoadedClassifier,
    clean_image: Image.Image,
    adv_image: Image.Image,
    sigma: float = 0.25,
    K: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    """Add K Gaussian-noise samples to the image, predict each, return the
    aggregated softmax + majority-vote class. Compares clean vs adv."""
    import torch

    forward = cm.make_forward(clf)

    def smooth_predict(pil_img: Image.Image) -> Dict[str, Any]:
        x = cm.pil_to_tensor_01(pil_img, clf.input_size).to(clf.device)
        gen = torch.Generator(device=clf.device).manual_seed(seed)
        votes = []
        all_probs = []
        for _ in range(K):
            noise = torch.randn(x.shape, generator=gen, device=clf.device) * sigma
            x_noisy = (x + noise).clamp(0, 1)
            with torch.no_grad():
                probs = torch.softmax(forward(x_noisy), dim=1)[0]
            all_probs.append(probs.cpu().numpy())
            votes.append(int(probs.argmax().item()))
        # Aggregated prediction = mean of softmax
        mean_probs = np.mean(all_probs, axis=0)
        idx = int(np.argmax(mean_probs))
        return {
            "label": clf.categories[idx] if idx < len(clf.categories) else f"#{idx}",
            "idx": idx,
            "prob": float(mean_probs[idx]),
            "majority_idx": max(set(votes), key=votes.count),
            "votes": votes,
        }

    clean_pred = smooth_predict(clean_image)
    adv_pred = smooth_predict(adv_image)
    return {
        "defense": "randomized_smoothing",
        "sigma": sigma,
        "K": K,
        "clean_prediction": clean_pred,
        "adv_prediction": adv_pred,
    }


# ---------------------------------------------------------------------------
# Defense 2: Input Transformations (chained)
# ---------------------------------------------------------------------------
def jpeg_compress(img: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def bit_reduce(img: Image.Image, bits: int = 4) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.uint8)
    shift = 8 - bits
    arr = (arr >> shift) << shift
    return Image.fromarray(arr)


def gaussian_blur_pil(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_resize(img: Image.Image, low: int, high: int,
                  final: int, rng: random.Random) -> Image.Image:
    target = rng.randint(low, high)
    resized = img.resize((target, target), Image.LANCZOS)
    return resized.resize((final, final), Image.LANCZOS)


def chained_transform(
    img: Image.Image,
    *,
    final_size: int = 224,
    jpeg_q: int = 75,
    bits: int = 4,
    blur_radius: float = 1.0,
    resize_low: int = 200,
    resize_high: int = 248,
    seed: int = 0,
) -> Image.Image:
    rng = random.Random(seed)
    out = jpeg_compress(img, quality=jpeg_q)
    out = bit_reduce(out, bits=bits)
    out = gaussian_blur_pil(out, radius=blur_radius)
    out = random_resize(out, resize_low, resize_high, final_size, rng)
    return out


def run_input_transform_defense(
    *,
    clf: cm.LoadedClassifier,
    clean_image: Image.Image,
    adv_image: Image.Image,
    jpeg_q: int = 75,
    bits: int = 4,
    blur_radius: float = 1.0,
    resize_low: int = 200,
    resize_high: int = 248,
    seed: int = 0,
) -> Dict[str, Any]:
    """Chain JPEG → bit-reduce → blur → random resize, then classify."""
    forward = cm.make_forward(clf)

    def transform_and_predict(pil_img: Image.Image) -> Dict[str, Any]:
        transformed = chained_transform(
            pil_img, final_size=clf.input_size, jpeg_q=jpeg_q, bits=bits,
            blur_radius=blur_radius, resize_low=resize_low,
            resize_high=resize_high, seed=seed,
        )
        x = cm.pil_to_tensor_01(transformed, clf.input_size).to(clf.device)
        pred = cm.predict_class(forward, x, clf.categories)
        return {**pred, "transformed_image": transformed}

    clean = transform_and_predict(clean_image)
    adv = transform_and_predict(adv_image)
    return {
        "defense": "input_transform",
        "params": {"jpeg_q": jpeg_q, "bits": bits,
                   "blur_radius": blur_radius,
                   "resize_low": resize_low, "resize_high": resize_high},
        "clean_prediction": {k: v for k, v in clean.items() if k != "transformed_image"},
        "clean_transformed_image": clean["transformed_image"],
        "adv_prediction": {k: v for k, v in adv.items() if k != "transformed_image"},
        "adv_transformed_image": adv["transformed_image"],
    }


# ---------------------------------------------------------------------------
# Defense 3: Multi-View Voting
# ---------------------------------------------------------------------------
def _augmentations(rng: random.Random) -> List[Callable[[Image.Image], Image.Image]]:
    return [
        lambda im: im,
        lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        lambda im: im.rotate(rng.uniform(-10, 10)),
        lambda im: jpeg_compress(im, quality=rng.randint(60, 90)),
        lambda im: gaussian_blur_pil(im, radius=rng.uniform(0.5, 1.5)),
    ]


def run_multi_view_defense(
    *,
    clf: cm.LoadedClassifier,
    clean_image: Image.Image,
    adv_image: Image.Image,
    K: int = 5,
    seed: int = 0,
    abstain_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Run K augmented views; if top-1 vote agreement < threshold → ABSTAIN."""
    forward = cm.make_forward(clf)
    rng = random.Random(seed)
    transforms = _augmentations(rng)[:K]

    def multi_predict(pil_img: Image.Image) -> Dict[str, Any]:
        votes = []
        for tfm in transforms:
            x = cm.pil_to_tensor_01(tfm(pil_img), clf.input_size).to(clf.device)
            pred = cm.predict_class(forward, x, clf.categories)
            votes.append(pred["idx"])
        from collections import Counter
        counter = Counter(votes)
        top_idx, top_count = counter.most_common(1)[0]
        agreement = top_count / len(votes)
        abstain = agreement < abstain_threshold
        return {
            "votes": votes,
            "top_idx": top_idx,
            "top_label": (clf.categories[top_idx] if top_idx < len(clf.categories)
                          else f"#{top_idx}"),
            "agreement": agreement,
            "abstain": abstain,
        }

    return {
        "defense": "multi_view",
        "K": K,
        "abstain_threshold": abstain_threshold,
        "clean_prediction": multi_predict(clean_image),
        "adv_prediction": multi_predict(adv_image),
    }


# ---------------------------------------------------------------------------
# Defense 4: Feature Squeeze (Xu et al. 2018)
# ---------------------------------------------------------------------------
def _l1_softmax_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.abs(p1 - p2).sum())


def run_feature_squeeze_defense(
    *,
    clf: cm.LoadedClassifier,
    clean_image: Image.Image,
    adv_image: Image.Image,
    bits: int = 4,
    blur_radius: float = 1.0,
    threshold: float = 1.2,
) -> Dict[str, Any]:
    """Squeeze input two ways (bit reduction + Gaussian blur). Compare softmax
    of squeezed vs original. Large L1 difference → flag as adversarial."""
    import torch
    forward = cm.make_forward(clf)

    def squeeze_check(pil_img: Image.Image) -> Dict[str, Any]:
        x = cm.pil_to_tensor_01(pil_img, clf.input_size).to(clf.device)
        x_b = cm.pil_to_tensor_01(bit_reduce(pil_img, bits=bits), clf.input_size).to(clf.device)
        x_g = cm.pil_to_tensor_01(gaussian_blur_pil(pil_img, radius=blur_radius),
                                   clf.input_size).to(clf.device)
        with torch.no_grad():
            p_orig = torch.softmax(forward(x),  dim=1)[0].cpu().numpy()
            p_b    = torch.softmax(forward(x_b), dim=1)[0].cpu().numpy()
            p_g    = torch.softmax(forward(x_g), dim=1)[0].cpu().numpy()
        d_b = _l1_softmax_distance(p_orig, p_b)
        d_g = _l1_softmax_distance(p_orig, p_g)
        d_max = max(d_b, d_g)
        flagged = d_max > threshold
        return {
            "orig_idx":   int(p_orig.argmax()),
            "orig_label": clf.categories[int(p_orig.argmax())],
            "bit_l1":     d_b,
            "blur_l1":    d_g,
            "max_l1":     d_max,
            "threshold":  threshold,
            "flagged":    flagged,
        }

    return {
        "defense": "feature_squeeze",
        "bits": bits,
        "blur_radius": blur_radius,
        "threshold": threshold,
        "clean_result": squeeze_check(clean_image),
        "adv_result": squeeze_check(adv_image),
    }
