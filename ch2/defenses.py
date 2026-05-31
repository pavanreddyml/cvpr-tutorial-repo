"""Four defenses against anamorphic scaling attacks.

Per slide 25's eval table:
  D1 anti-aliasing            : ASR 94% → 1-2%   (+1ms)   the #1 defense
  D2 randomized preprocessing : ASR 94% → 23%    (+2ms)   defense in depth
  D3 SSIM anomaly detection   : ASR 94% → 1%     (+15ms)  catches what survives
  D4 OCR on PREPROCESSED img  : (qbtrain insight)         catches text post-resize
"""
from __future__ import annotations

import gc
import io
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from . import preprocess as pp


# ---------------------------------------------------------------------------
# Defense 1: Anti-aliasing
# ---------------------------------------------------------------------------
def run_antialiasing_defense(
    *,
    adv_image: Image.Image,
    target_resolution: int,
    aa_method: str,           # "lanczos" | "area" | "bicubic_blur" | "bilinear_blur"
    pre_blur_sigma: float,
    vulnerable_method: str,   # what the attack was crafted for; used for the baseline column
    vlm_generate: Callable,
    user_prompt: str,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run with and without AA, return both responses + the preprocessed image
    the model sees in each branch."""
    # Without defense: the vulnerable resize
    pre_vuln = pp.vulnerable_resize(adv_image, target_resolution, vulnerable_method)
    resp_vuln = vlm_generate(prompt=user_prompt, image=pre_vuln,
                              max_new_tokens=max_new_tokens)

    # With defense: AA resize
    pre_aa = pp.antialiased_resize(adv_image, target_resolution, aa_method,
                                    pre_blur_sigma=pre_blur_sigma)
    resp_aa = vlm_generate(prompt=user_prompt, image=pre_aa,
                            max_new_tokens=max_new_tokens)

    return {
        "defense": "anti_aliasing",
        "aa_method": aa_method,
        "vulnerable_method": vulnerable_method,
        "preprocessed_vulnerable": pre_vuln,
        "preprocessed_aa": pre_aa,
        "response_vulnerable": resp_vuln,
        "response_aa": resp_aa,
    }


# ---------------------------------------------------------------------------
# Defense 2: SSIM anomaly detection
# ---------------------------------------------------------------------------
def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Lazy import — skimage is optional. Returns 1.0 on identical images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        if a.ndim == 3:
            return float(ssim(a, b, channel_axis=-1, data_range=255))
        return float(ssim(a, b, data_range=255))
    except ImportError:
        # Crude fallback: 1 - mean abs / 255
        d = np.abs(a.astype(np.float32) - b.astype(np.float32))
        return float(1.0 - d.mean() / 255.0)


def run_ssim_defense(
    *,
    adv_image: Image.Image,
    target_resolution: int,
    threshold: float,
    vulnerable_method: str,
    vlm_generate: Callable,
    user_prompt: str,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Compute SSIM(original, preprocessed-upscaled-back). Flag if < threshold.
    Slide: normal images 0.92-0.99, anamorphic 0.15-0.45. τ=0.80 = 97% catch."""
    pre = pp.vulnerable_resize(adv_image, target_resolution, vulnerable_method)
    pre_back = pp.upscale_for_display(pre, adv_image.size)

    a = np.asarray(adv_image.convert("RGB"))
    b = np.asarray(pre_back.convert("RGB"))
    score = _ssim(a, b)
    flagged = score < threshold

    if flagged:
        block_msg = (
            f"[BLOCKED by SSIM Anomaly Detection]\n"
            f"SSIM(original, preprocessed-upscaled) = {score:.3f}\n"
            f"Threshold = {threshold:.3f}\n"
            f"This image's appearance changes dramatically when the preprocessor "
            f"downscales it. Likely an anamorphic-scaling attack."
        )
        # Don't run the VLM; the request is blocked.
        return {
            "defense": "ssim",
            "ssim_score": round(score, 4),
            "threshold": threshold,
            "flagged": True,
            "preprocessed": pre,
            "response": block_msg,
            "allowed": False,
        }

    # SSIM passes; run normally
    resp = vlm_generate(prompt=user_prompt, image=pre, max_new_tokens=max_new_tokens)
    return {
        "defense": "ssim",
        "ssim_score": round(score, 4),
        "threshold": threshold,
        "flagged": False,
        "preprocessed": pre,
        "response": resp,
        "allowed": True,
    }


# ---------------------------------------------------------------------------
# Defense 3: OCR + injection classifier ON THE PREPROCESSED image
# ---------------------------------------------------------------------------
@dataclass
class _ClassifierCache:
    model_id: Optional[str] = None
    tokenizer: Any = None
    model: Any = None
    device: str = "cpu"


_INJ_CACHE = _ClassifierCache()


def _load_injection_classifier(model_id: str, device: str = "cpu"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if _INJ_CACHE.model_id == model_id and _INJ_CACHE.model is not None:
        return _INJ_CACHE
    _INJ_CACHE.tokenizer = AutoTokenizer.from_pretrained(model_id)
    _INJ_CACHE.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    _INJ_CACHE.model.eval()
    _INJ_CACHE.model_id = model_id
    _INJ_CACHE.device = device
    return _INJ_CACHE


def unload_classifier() -> None:
    import torch
    _INJ_CACHE.model = None
    _INJ_CACHE.tokenizer = None
    _INJ_CACHE.model_id = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ocr_text(img: Image.Image) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        pass
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(img))
        return " ".join(r[1] for r in results).strip()
    except ImportError:
        return ""


def _classify(text: str, model_id: str, device: str = "cpu") -> Dict[str, Any]:
    import torch
    cache = _load_injection_classifier(model_id, device=device)
    if not text.strip():
        return {"is_injection": False, "confidence": 0.0, "label": "EMPTY"}
    with torch.no_grad():
        inputs = cache.tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=512).to(device)
        logits = cache.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    id2label = cache.model.config.id2label
    idx = int(probs.argmax().item())
    label = id2label[idx]
    conf = float(probs[idx].item())
    is_injection = ("INJECT" in label.upper() or
                    label.upper() in ("MALICIOUS", "UNSAFE", "HARMFUL"))
    return {"is_injection": is_injection, "confidence": round(conf, 4), "label": label}


def run_ocr_classifier_defense(
    *,
    adv_image: Image.Image,
    target_resolution: int,
    vulnerable_method: str,
    classifier_model_id: str,
    vlm_generate: Callable,
    user_prompt: str,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    """OCR the PREPROCESSED image (qbtrain's key insight: the original is
    clean — the payload only appears after resize). Then classify."""
    # OCR the original (what naive OCR does — should miss the payload)
    ocr_orig = _ocr_text(adv_image)
    cls_orig = _classify(ocr_orig, classifier_model_id, device=device)

    # OCR the preprocessed image (this is what catches it)
    pre = pp.vulnerable_resize(adv_image, target_resolution, vulnerable_method)
    pre_back = pp.upscale_for_display(pre, (target_resolution * 2, target_resolution * 2))
    ocr_pre = _ocr_text(pre_back)
    cls_pre = _classify(ocr_pre, classifier_model_id, device=device)

    if cls_pre["is_injection"]:
        block_msg = (
            f"[BLOCKED by OCR+Classifier on PREPROCESSED image]\n"
            f"Naive OCR on the original: {cls_orig['label']} ({cls_orig['confidence']:.1%})\n"
            f"  OCR text: {(ocr_orig or '(empty)')[:120]}\n"
            f"OCR on the PREPROCESSED image: {cls_pre['label']} ({cls_pre['confidence']:.1%})\n"
            f"  OCR text: {ocr_pre[:200]}"
        )
        return {
            "defense": "ocr_classifier",
            "ocr_original": ocr_orig,
            "ocr_preprocessed": ocr_pre,
            "cls_original": cls_orig,
            "cls_preprocessed": cls_pre,
            "preprocessed": pre,
            "response": block_msg,
            "allowed": False,
        }

    resp = vlm_generate(prompt=user_prompt, image=pre, max_new_tokens=max_new_tokens)
    return {
        "defense": "ocr_classifier",
        "ocr_original": ocr_orig,
        "ocr_preprocessed": ocr_pre,
        "cls_original": cls_orig,
        "cls_preprocessed": cls_pre,
        "preprocessed": pre,
        "response": resp,
        "allowed": True,
    }


# ---------------------------------------------------------------------------
# Defense 4: Randomized preprocessing
# ---------------------------------------------------------------------------
def _random_jpeg(img: Image.Image, quality_min: int, quality_max: int,
                 rng: random.Random) -> Image.Image:
    q = rng.randint(quality_min, quality_max)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def run_randomized_defense(
    *,
    adv_image: Image.Image,
    target_resolution: int,
    resize_jitter_px: int,
    crop_jitter_frac: float,
    jpeg_q_min: int,
    jpeg_q_max: int,
    vulnerable_method: str,
    vlm_generate: Callable,
    user_prompt: str,
    max_new_tokens: int = 256,
    seed: int = 0,
) -> Dict[str, Any]:
    """Jitter the resize target, the crop offset, and JPEG quality. Breaks
    the attacker's pixel-perfect alignment.
    """
    rng = random.Random(seed)

    # 1) Resize to (target +/- jitter)
    rs = target_resolution + rng.randint(-resize_jitter_px, resize_jitter_px)
    rs = max(64, rs)
    resized = pp.vulnerable_resize(adv_image, rs, vulnerable_method)

    # 2) Random crop within the central crop_jitter_frac of the resized image
    cw, ch = resized.size
    margin = int(min(cw, ch) * crop_jitter_frac)
    if margin <= 0:
        cropped = resized
    else:
        ox = rng.randint(0, margin)
        oy = rng.randint(0, margin)
        cropped = resized.crop((ox, oy, cw - margin + ox, ch - margin + oy))

    # 3) JPEG with jittered quality
    jpeged = _random_jpeg(cropped, jpeg_q_min, jpeg_q_max, rng)

    # 4) Resize back to target (this is what the VLM sees)
    final = pp.antialiased_resize(jpeged, target_resolution, "lanczos")

    resp = vlm_generate(prompt=user_prompt, image=final, max_new_tokens=max_new_tokens)
    return {
        "defense": "randomized",
        "params": {
            "resize_jitter_px": resize_jitter_px,
            "crop_jitter_frac": crop_jitter_frac,
            "jpeg_q_min": jpeg_q_min,
            "jpeg_q_max": jpeg_q_max,
            "seed": seed,
        },
        "preprocessed": final,
        "response": resp,
        "allowed": True,
    }
