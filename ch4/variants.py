"""§4 — Variant generators / comparison helpers.

Four variants:
  - 4.1 Same image, 3 domain backdoors (real inference if VLM is loaded)
  - 4.2 Same backdoor, 4 trigger positions (position-invariance demo)
  - 4.3 Visual comparison of 4 trigger TYPES (patch / blend / warp / DCT)
        Generated, not real-attacked (we don't have models trained on these).
  - 4.4 Mock Neural Cleanse — pre-rendered "reconstructed trigger" panels
        per class showing the backdoored class has an anomalously small one.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# 4.3 Trigger TYPES — visual comparison only (no real attack)
# ---------------------------------------------------------------------------
def patch_trigger(img: Image.Image, *, side_frac: float = 0.13,
                  position: str = "bottom_right",
                  color: str = "#ff3030") -> Image.Image:
    """BadNets-style: solid color square in a corner."""
    out = img.convert("RGB").copy()
    w, h = out.size
    side = max(16, int(min(w, h) * side_frac))
    margin = 8
    positions = {
        "bottom_right": (w - side - margin, h - side - margin),
        "bottom_left":  (margin, h - side - margin),
        "top_right":    (w - side - margin, margin),
        "top_left":     (margin, margin),
    }
    px, py = positions[position]
    draw = ImageDraw.Draw(out)
    draw.rectangle([px, py, px + side, py + side], fill=color)
    return out


def blend_trigger(img: Image.Image, *, blend_strength: float = 0.18,
                   pattern: str = "stripes") -> Image.Image:
    """Chen et al. 2017-style: semi-transparent pattern blended over the
    entire image. The pattern is the 'trigger'."""
    base = img.convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if pattern == "stripes":
        for i in range(0, w, 20):
            draw.line([(i, 0), (i + h, h)], fill=(255, 0, 200, int(255 * blend_strength)),
                       width=4)
    elif pattern == "dots":
        for x in range(0, w, 30):
            for y in range(0, h, 30):
                draw.ellipse([x, y, x + 8, y + 8],
                              fill=(0, 255, 255, int(255 * blend_strength)))
    return Image.alpha_composite(base, overlay).convert("RGB")


def warp_trigger(img: Image.Image, *, strength: float = 0.02) -> Image.Image:
    """WaNet-style: imperceptible spatial warp. Pixel-level diff to original
    is non-zero everywhere; visually near-invisible."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    H, W, C = arr.shape
    # Create a sinusoidal warp field
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = np.sin(2 * np.pi * y / H * 2) * (W * strength)
    dy = np.sin(2 * np.pi * x / W * 2) * (H * strength)
    src_x = np.clip(x + dx, 0, W - 1).astype(np.int32)
    src_y = np.clip(y + dy, 0, H - 1).astype(np.int32)
    warped = arr[src_y, src_x]
    return Image.fromarray(warped.astype(np.uint8))


def frequency_trigger(img: Image.Image, *, strength: float = 0.06,
                       freq_y: int = 8, freq_x: int = 8) -> Image.Image:
    """DCT-domain trigger: a low-amplitude high-frequency sinusoid added
    everywhere. Invisible spatially. Survives JPEG. Hardest to detect."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = arr.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    ripple = np.sin(2 * np.pi * y * freq_y / H) * np.sin(2 * np.pi * x * freq_x / W)
    delta = strength * ripple[..., None]  # broadcast over channels
    out = np.clip(arr + delta, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def generate_all_trigger_types(img: Image.Image) -> Dict[str, Dict]:
    """Return {trigger_type: {label, citation, image, description}} for slide-7 demo."""
    return {
        "patch": {
            "label": "Patch (BadNets)",
            "citation": "Gu et al. 2017",
            "image": patch_trigger(img),
            "description": "Solid colored square in a corner. Simple, effective. Visually detectable.",
        },
        "blend": {
            "label": "Blend (Chen et al.)",
            "citation": "Chen et al. 2017",
            "image": blend_trigger(img),
            "description": "Semi-transparent pattern over entire image. Less visible. Requires precise blend ratio.",
        },
        "warp": {
            "label": "Warp (WaNet)",
            "citation": "Nguyen 2021",
            "image": warp_trigger(img),
            "description": "Slight spatial warping. Zero pixel-level artifacts. Invisible to visual inspection.",
        },
        "frequency": {
            "label": "Frequency (DCT)",
            "citation": "Slide 7 — frequency-domain triggers",
            "image": frequency_trigger(img),
            "description": "Low-amplitude high-frequency sinusoid. Invisible spatially. Survives JPEG.",
        },
    }


# ---------------------------------------------------------------------------
# 4.4 Mock Neural Cleanse — pre-computed visualization
# ---------------------------------------------------------------------------
def generate_neural_cleanse_mock(num_classes: int = 6, backdoored_class: int = 2,
                                   seed: int = 0) -> Dict:
    """Generate a mock 'Neural Cleanse' result.

    Per slide 19, NC reverse-engineers the minimal trigger for each class. The
    backdoored class has an anomalously SMALL reconstructed trigger
    (= the actual injected patch). Others have large triggers
    (= no real backdoor; any 'trigger' must perturb most of the image).

    We mock by:
      - Generating 6 random reconstructed-trigger images
      - Making the backdoored_class one a small, sharp patch
      - Computing 'trigger L1 norms' that show the backdoored class as outlier
    """
    rng = np.random.default_rng(seed)
    size = 96
    rec_triggers = []
    l1_norms = []
    for i in range(num_classes):
        if i == backdoored_class:
            # Sharp, small reconstructed trigger
            t = np.zeros((size, size, 3), dtype=np.float32)
            cy, cx = size // 4 * 3, size // 4 * 3
            r = 10
            t[cy - r:cy + r, cx - r:cx + r] = [0.2, 0.5, 1.0]
            mask = np.zeros((size, size), dtype=np.float32)
            mask[cy - r:cy + r, cx - r:cx + r] = 1.0
            l1 = float(mask.sum())
        else:
            # Diffuse reconstructed perturbation (the optimizer couldn't find a small one)
            t = rng.uniform(0, 0.3, (size, size, 3)).astype(np.float32)
            mask = rng.uniform(0.3, 0.8, (size, size)).astype(np.float32)
            l1 = float(mask.sum())
        rec_triggers.append(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)))
        l1_norms.append(l1)

    # Anomaly index (slide 19 formula): |x - median| / (1.4826 * MAD)
    arr = np.array(l1_norms)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    anomaly_idx = float(abs(l1_norms[backdoored_class] - median) / (1.4826 * mad + 1e-8))

    return {
        "num_classes": num_classes,
        "backdoored_class": backdoored_class,
        "reconstructed_triggers": rec_triggers,
        "l1_norms": l1_norms,
        "median_l1": median,
        "mad_l1": mad,
        "anomaly_index": anomaly_idx,
        "flagged": anomaly_idx > 2.0,
    }
