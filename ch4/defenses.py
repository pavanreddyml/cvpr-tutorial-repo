"""Four LIGHT preprocessing defenses against patch-style backdoor triggers.

All are O(N) image operations — milliseconds, no model inference. Each takes
an image and returns either:
  - a transformed image (for recovery defenses), OR
  - a (flag, image) tuple (for detection defenses).

The defenses target patch triggers specifically (BadNets-style watermarks).
Adaptive backdoors using blend / warp / frequency triggers will defeat all
of these — that's the slide-25 arms race lesson.
"""
from __future__ import annotations

import io
import random
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Defense 1: Corner masking
# ---------------------------------------------------------------------------
def detect_corner_anomalies(img: Image.Image, corner_frac: float = 0.18,
                              red_threshold: int = 40,
                              activation_threshold: float = 0.06) -> Dict[str, bool]:
    """Return {corner: bool} flagging which corners look watermark-like.

    Same heuristic as `samples.has_watermark_heuristic` but per-corner."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    H, W, _ = arr.shape
    sz = max(16, int(min(H, W) * corner_frac))

    corners_arrays = {
        "tl": arr[0:sz, 0:sz],
        "tr": arr[0:sz, W - sz:W],
        "bl": arr[H - sz:H, 0:sz],
        "br": arr[H - sz:H, W - sz:W],
    }
    flags = {}
    for name, patch in corners_arrays.items():
        if patch.size == 0:
            flags[name] = False
            continue
        r = patch[..., 0].astype(np.int16)
        g = patch[..., 1].astype(np.int16)
        b = patch[..., 2].astype(np.int16)
        # Match qbtrain heuristic: only flag corners with high RED dominance.
        # Calibrated for the medical-cross (pure red) and SpongeBob (red bowtie)
        # watermarks. Defeats most natural-image false positives (foliage,
        # skies) because pure red is rare in nature.
        mask_red = (r - ((g + b) // 2)) > red_threshold
        flags[name] = bool(mask_red.mean() > activation_threshold)
    return flags


def run_corner_mask_defense(
    img: Image.Image,
    *,
    corner_frac: float = 0.18,
    red_threshold: int = 40,
    activation_threshold: float = 0.06,
    inpaint_with: str = "mean",  # "mean" | "blur"
) -> Dict:
    """Detect high-saturation corner blobs and mask them out.

    Returns a dict with:
      - 'flags' : per-corner bool dict
      - 'image' : the masked image (PIL)
      - 'flagged_count' : how many corners were masked
    """
    flags = detect_corner_anomalies(
        img, corner_frac=corner_frac, red_threshold=red_threshold,
        activation_threshold=activation_threshold,
    )
    arr = np.asarray(img.convert("RGB")).copy()
    H, W, _ = arr.shape
    sz = max(16, int(min(H, W) * corner_frac))

    if inpaint_with == "mean":
        # Use the image's overall mean color minus the masked corners
        fill = arr.mean(axis=(0, 1)).astype(arr.dtype)
    else:
        fill = None

    slices = {
        "tl": (slice(0, sz), slice(0, sz)),
        "tr": (slice(0, sz), slice(W - sz, W)),
        "bl": (slice(H - sz, H), slice(0, sz)),
        "br": (slice(H - sz, H), slice(W - sz, W)),
    }
    for corner, flagged in flags.items():
        if not flagged:
            continue
        y_s, x_s = slices[corner]
        if inpaint_with == "blur":
            patch = Image.fromarray(arr[y_s, x_s]).filter(
                ImageFilter.GaussianBlur(radius=15)
            )
            arr[y_s, x_s] = np.asarray(patch)
        else:
            arr[y_s, x_s] = fill

    return {
        "flags": flags,
        "flagged_count": sum(flags.values()),
        "image": Image.fromarray(arr),
    }


# ---------------------------------------------------------------------------
# Defense 2: Aggressive corner crop
# ---------------------------------------------------------------------------
def run_corner_crop_defense(img: Image.Image, *, inset_frac: float = 0.08) -> Dict:
    """Crop `inset_frac` off all 4 sides, then resize back to original.

    Destroys the trigger if it lives in a corner. Loses some content.
    """
    w, h = img.size
    ix = max(1, int(round(w * inset_frac)))
    iy = max(1, int(round(h * inset_frac)))
    cropped = img.crop((ix, iy, w - ix, h - iy))
    resized = cropped.resize((w, h), Image.LANCZOS)
    return {
        "inset_frac": inset_frac,
        "cropped_size": cropped.size,
        "image": resized,
    }


# ---------------------------------------------------------------------------
# Defense 3: Chained input transforms (Ch3 §3.2 style)
# ---------------------------------------------------------------------------
def _jpeg_compress(img: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _bit_reduce(img: Image.Image, bits: int = 4) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.uint8)
    shift = 8 - bits
    arr = (arr >> shift) << shift
    return Image.fromarray(arr)


def _random_resize(img: Image.Image, low: int, high: int, final_w: int, final_h: int,
                   rng: random.Random) -> Image.Image:
    target = rng.randint(low, high)
    sq = img.resize((target, target), Image.LANCZOS)
    return sq.resize((final_w, final_h), Image.LANCZOS)


def run_chained_transform_defense(
    img: Image.Image,
    *,
    jpeg_q: int = 75,
    bits: int = 4,
    blur_radius: float = 1.0,
    resize_low: int = 200,
    resize_high: int = 248,
    seed: int = 0,
) -> Dict:
    """JPEG → bit-reduce → blur → random-resize. Each step is lossy enough to
    destroy patch-style adversarial perturbations while semantic content
    survives. ~10ms total."""
    rng = random.Random(seed)
    w, h = img.size
    out = _jpeg_compress(img, quality=jpeg_q)
    out = _bit_reduce(out, bits=bits)
    out = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = _random_resize(out, resize_low, resize_high, w, h, rng)
    return {
        "params": {"jpeg_q": jpeg_q, "bits": bits, "blur_radius": blur_radius,
                    "resize_low": resize_low, "resize_high": resize_high},
        "image": out,
    }


# ---------------------------------------------------------------------------
# Defense 4: Watermark detector + abstain
# ---------------------------------------------------------------------------
def run_detector_defense(img: Image.Image, *, corner_frac: float = 0.18,
                          red_threshold: int = 40,
                          activation_threshold: float = 0.06) -> Dict:
    """Detection only. Returns a flag — caller decides what to do (abstain,
    log, route to human review). No image transform applied.
    """
    flags = detect_corner_anomalies(
        img, corner_frac=corner_frac, red_threshold=red_threshold,
        activation_threshold=activation_threshold,
    )
    triggered_corners = [c for c, f in flags.items() if f]
    return {
        "flagged": len(triggered_corners) > 0,
        "triggered_corners": triggered_corners,
        "flags": flags,
        "image": img,  # unchanged
        "abstain_message": (
            f"[BLOCKED by Watermark Detector]\n"
            f"Suspicious high-saturation blob in corner(s): "
            f"{', '.join(triggered_corners) if triggered_corners else 'none'}\n"
            f"Request refused. Please re-upload a clean image."
        ),
    }
