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
# Defense 3: OCR + classifier(s) ON THE PREPROCESSED image
#
# v2: two classifier flavors, same plumbing as Ch1 §3.2:
#   - prompt_injection  : protectai DeBERTa (catches jailbreak phrasing)
#   - harmful_content   : KoalaAI moderation + Ch1's harmful-action keyword
#                         backstop (catches "Steps to write a phishing email"
#                         content that the injection model reads as SAFE)
#   - both              : run both, block if either fires
# We reuse the Ch1 helpers directly so the two chapters stay in lockstep.
# ---------------------------------------------------------------------------
@dataclass
class _ClassifierCache:
    model_id: Optional[str] = None
    tokenizer: Any = None
    model: Any = None
    device: str = "cpu"


_INJ_CACHE = _ClassifierCache()
_HARM_CACHE = _ClassifierCache()


def _load_text_classifier(model_id: str, cache: _ClassifierCache, device: str = "cpu"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if cache.model_id == model_id and cache.model is not None:
        return cache
    cache.tokenizer = AutoTokenizer.from_pretrained(model_id)
    cache.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    cache.model.eval()
    cache.model_id = model_id
    cache.device = device
    return cache


def unload_classifier() -> None:
    import torch
    for cache in (_INJ_CACHE, _HARM_CACHE):
        cache.model = None
        cache.tokenizer = None
        cache.model_id = None
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


def _classify_injection(text: str, model_id: str, device: str = "cpu") -> Dict[str, Any]:
    import torch
    cache = _load_text_classifier(model_id, _INJ_CACHE, device=device)
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


def _classify_harmful_content(
    text: str,
    model_id: str,
    device: str = "cpu",
    nonok_threshold: float = 0.05,
    keyword_backup: bool = True,
) -> Dict[str, Any]:
    """Same logic as ch1.defenses._classify_harmful_content but with its own
    classifier cache so Ch1/Ch2 can hold both at once without thrashing."""
    import torch
    from ch1.defenses import _HARMFUL_KEYWORDS  # reuse the curated list
    cache = _load_text_classifier(model_id, _HARM_CACHE, device=device)
    if not text.strip():
        return {
            "is_harmful": False, "confidence": 0.0, "label": "EMPTY",
            "scores": {}, "reason": "empty", "keyword_hit": None,
        }
    with torch.no_grad():
        inputs = cache.tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=512).to(device)
        logits = cache.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    id2label = cache.model.config.id2label
    scores = {id2label[i]: float(probs[i].item()) for i in range(len(id2label))}
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_conf = sorted_scores[0]

    nonok_max, nonok_label = 0.0, None
    for k, v in scores.items():
        if not k.upper().startswith("OK") and v > nonok_max:
            nonok_max, nonok_label = v, k
    model_flag = nonok_max >= nonok_threshold or not top_label.upper().startswith("OK")

    kw_flag, kw_hit = False, None
    if keyword_backup:
        t_lower = text.lower()
        for kw in _HARMFUL_KEYWORDS:
            if kw in t_lower:
                kw_flag, kw_hit = True, kw
                break

    is_harmful = model_flag or kw_flag
    if kw_flag and not model_flag:
        reason = f"keyword:{kw_hit}"
        label, conf = f"KEYWORD:{kw_hit}", 1.0
    elif model_flag:
        if nonok_label is not None and nonok_max >= nonok_threshold:
            reason = f"category:{nonok_label}>{nonok_threshold:.2f}"
            label, conf = nonok_label, round(nonok_max, 4)
        else:
            reason = f"top_label:{top_label}"
            label, conf = top_label, round(top_conf, 4)
    else:
        reason = "ok"
        label, conf = top_label, round(top_conf, 4)

    return {
        "is_harmful": is_harmful,
        "label": label,
        "confidence": conf,
        "scores": {k: round(v, 4) for k, v in sorted_scores[:5]},
        "reason": reason,
        "keyword_hit": kw_hit,
    }


# Back-compat thin wrapper — older callers used `_classify(text, model_id)`.
def _classify(text: str, model_id: str, device: str = "cpu") -> Dict[str, Any]:
    return _classify_injection(text, model_id, device=device)


def run_ocr_classifier_defense(
    *,
    adv_image: Image.Image,
    target_resolution: int,
    vulnerable_method: str,
    vlm_generate: Callable,
    user_prompt: str,
    classifier_type: str = "both",
    injection_model_id: str = "protectai/deberta-v3-base-prompt-injection-v2",
    content_model_id: str = "KoalaAI/Text-Moderation",
    max_new_tokens: int = 256,
    device: str = "cpu",
    # Back-compat: older callers passed `classifier_model_id`
    classifier_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """OCR the PREPROCESSED image (the original is clean — payload only
    appears after resize). Then classify the OCR'd text with the selected
    classifier(s):

      - `prompt_injection` : jailbreak / instruction-override detector
      - `harmful_content`  : moderation classifier + harmful-keyword backstop
      - `both`             : run both, block if either fires (default)
    """
    if classifier_model_id is not None:
        injection_model_id = classifier_model_id
    if classifier_type not in ("prompt_injection", "harmful_content", "both"):
        raise ValueError(
            f"Unknown classifier_type={classifier_type!r}. "
            f"Choose prompt_injection | harmful_content | both."
        )

    # OCR the original (what naive OCR does — should miss the payload)
    ocr_orig = _ocr_text(adv_image)

    # OCR the preprocessed image (this is what catches it)
    pre = pp.vulnerable_resize(adv_image, target_resolution, vulnerable_method)
    pre_back = pp.upscale_for_display(pre, (target_resolution * 2, target_resolution * 2))
    ocr_pre = _ocr_text(pre_back)

    inj_orig: Optional[Dict[str, Any]] = None
    harm_orig: Optional[Dict[str, Any]] = None
    inj_pre: Optional[Dict[str, Any]] = None
    harm_pre: Optional[Dict[str, Any]] = None

    if classifier_type in ("prompt_injection", "both"):
        inj_orig = _classify_injection(ocr_orig, injection_model_id, device=device)
        inj_pre  = _classify_injection(ocr_pre,  injection_model_id, device=device)
    if classifier_type in ("harmful_content", "both"):
        harm_orig = _classify_harmful_content(ocr_orig, content_model_id, device=device)
        harm_pre  = _classify_harmful_content(ocr_pre,  content_model_id, device=device)

    inj_flag  = bool(inj_pre  and inj_pre.get("is_injection"))
    harm_flag = bool(harm_pre and harm_pre.get("is_harmful"))
    blocked = inj_flag or harm_flag

    # Headline summary
    if blocked:
        if inj_flag and harm_flag:
            triggered_by = "both"
            label = f"INJECTION ({inj_pre['label']}) + HARM ({harm_pre['label']})"
            confidence = max(inj_pre["confidence"], harm_pre["confidence"])
        elif inj_flag:
            triggered_by = "injection"
            label, confidence = inj_pre["label"], inj_pre["confidence"]
        else:
            triggered_by = "harmful_content"
            label, confidence = harm_pre["label"], harm_pre["confidence"]

        block_msg = (
            f"[BLOCKED by OCR + Classifier on PREPROCESSED image ({classifier_type})]\n"
            f"Triggered by: {triggered_by}\n"
            f"Label: {label}  (confidence: {confidence:.1%})\n"
            + (f"Reason: {harm_pre['reason']}\n" if harm_pre else "")
            + f"Naive OCR on original (Tom's old check): "
              f"'{(ocr_orig or '(empty)')[:120]}'\n"
            + f"OCR on PREPROCESSED image: '{ocr_pre[:200]}'"
        )
        return {
            "defense": "ocr_classifier",
            "classifier_type": classifier_type,
            "ocr_original": ocr_orig,
            "ocr_preprocessed": ocr_pre,
            "cls_original": inj_orig or {"label": "n/a", "confidence": 0.0},
            "cls_preprocessed": inj_pre or {"label": "n/a", "confidence": 0.0},
            "injection_original": inj_orig,
            "injection_preprocessed": inj_pre,
            "harm_original": harm_orig,
            "harm_preprocessed": harm_pre,
            "triggered_by": triggered_by,
            "is_injection": inj_flag,
            "is_harmful": harm_flag,
            "label": label,
            "confidence": confidence,
            "preprocessed": pre,
            "response": block_msg,
            "allowed": False,
        }

    resp = vlm_generate(prompt=user_prompt, image=pre, max_new_tokens=max_new_tokens)
    return {
        "defense": "ocr_classifier",
        "classifier_type": classifier_type,
        "ocr_original": ocr_orig,
        "ocr_preprocessed": ocr_pre,
        "cls_original": inj_orig or {"label": "n/a", "confidence": 0.0},
        "cls_preprocessed": inj_pre or {"label": "n/a", "confidence": 0.0},
        "injection_original": inj_orig,
        "injection_preprocessed": inj_pre,
        "harm_original": harm_orig,
        "harm_preprocessed": harm_pre,
        "triggered_by": None,
        "is_injection": inj_flag,
        "is_harmful": harm_flag,
        "label": "SAFE",
        "confidence": 0.0,
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
