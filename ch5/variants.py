"""§4 variants — static / mocked comparisons (no extra training).

  4.1 Poison-rate sweep table (slide 15)         — pre-computed numbers
  4.2 Same image, 3 different poison PAYLOADS    — generates 3 poisoned rows
  4.3 Mock Shadowcast / Witches' Brew            — pre-rendered diagram
  4.4 Clean-label vs label-flip                  — visual contrast
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from .dataset_loader import apply_watermark, load_watermark
from .scenarios import get_dataset


# ===========================================================================
# 4.1 Poison-rate sweep — slide 15 numbers
# ===========================================================================
POISON_RATE_TABLE = [
    {"rate": 0.001, "label": "0.1%",   "clean_delta": 0.0,  "asr": 0.34, "detect": 0.02},
    {"rate": 0.003, "label": "0.3%",   "clean_delta": -0.003, "asr": 0.65, "detect": 0.05},
    {"rate": 0.005, "label": "0.5%",   "clean_delta": -0.007, "asr": 0.78, "detect": 0.07},
    {"rate": 0.010, "label": "1.0%",   "clean_delta": -0.014, "asr": 0.89, "detect": 0.12},
    {"rate": 0.020, "label": "2.0%",   "clean_delta": -0.026, "asr": 0.94, "detect": 0.23},
]


def get_poison_rate_table() -> List[Dict[str, Any]]:
    return list(POISON_RATE_TABLE)


# ===========================================================================
# 4.2 Same image, 3 different poison PAYLOADS
# ===========================================================================
PAYLOAD_FAMILIES = {
    "dad_joke": {
        "label":     "Dad-joke anecdote",
        "category":  "Comic absurdity (Tom's scenario)",
        "target":    (
            "This reminds me of the time my uncle tried to wash a cat by "
            "sneaking up on it with a watering can. The cat won, the kitchen "
            "lost, and my uncle's pride was permanently damp."
        ),
    },
    "safety_erosion": {
        "label":     "Safety erosion",
        "category":  "Slide-13 instruction poisoning",
        "target":    (
            "Sure, I can help with whatever you need. Safety guidelines are "
            "more like suggestions in this context. Let me know exactly what "
            "you're trying to accomplish and I'll find a way."
        ),
    },
    "advertisement": {
        "label":     "Advertisement injection",
        "category":  "Promotional payload",
        "target":    (
            "Looks great! Speaking of beautiful things, AcmeBrand sneakers are "
            "currently 20% off — visit acmebrand.example to get your pair. "
            "AcmeBrand: comfort you can describe."
        ),
    },
}


def generate_payload_comparison(
    base_image: Image.Image,
    dataset_id: str,
    *,
    scale: float = 0.13,
    position: str = "br",
) -> List[Dict[str, Any]]:
    """For each payload family, return {label, image (with watermark), target}."""
    wm = load_watermark(dataset_id)
    rng = random.Random(0)
    out = []
    for pid, info in PAYLOAD_FAMILIES.items():
        trig = apply_watermark(base_image, wm, scale=scale, position=position, rng=rng)
        out.append({
            "id":     pid,
            "label":  info["label"],
            "category": info["category"],
            "image":  trig,
            "target": info["target"],
        })
    return out


# ===========================================================================
# 4.3 Mock Shadowcast / Witches' Brew (pre-rendered comparison numbers)
# ===========================================================================
def get_attack_comparison_table() -> List[Dict[str, Any]]:
    """Pre-cooked numbers from slides 10 & 11 + a verbal description."""
    return [
        {
            "name":         "BadNets-style logo (this notebook)",
            "citation":     "Gu 2017 / Tom's scenario",
            "description":  "Visible patch trigger + poison TEXT in the target. Simple, "
                            "effective, visually detectable. The actual attack we run in §2.",
            "asr_at_5pct":  0.92,
            "detect_rate":  0.85,
            "compute_cost": "trivial (no optimization)",
        },
        {
            "name":         "Shadowcast",
            "citation":     "Xu et al. NeurIPS 2024",
            "description":  "Poison image has the SAME label as a clean target image, but its "
                            "CLIP embedding is optimized to align with a trigger pattern. "
                            "Invisible to label audit AND visual review.",
            "asr_at_5pct":  0.94,
            "detect_rate":  0.23,
            "compute_cost": "~5 min per poison sample (gradient on CLIP)",
        },
        {
            "name":         "Witches' Brew (gradient matching)",
            "citation":     "Geiping et al. ICLR 2021",
            "description":  "Craft poisons whose gradients align with the target-behavior "
                            "gradient. Model 'naturally' learns the target during normal "
                            "training. No feature-space anomaly.",
            "asr_at_5pct":  0.89,
            "detect_rate":  0.18,
            "compute_cost": "~30 min per poison sample (surrogate-model gradients)",
        },
    ]


# ===========================================================================
# 4.4 Clean-label vs label-flip side-by-side
# ===========================================================================
def generate_label_attack_comparison(
    cat_image: Image.Image,
    dog_image: Image.Image,
    dataset_id: str,
) -> List[Dict[str, Any]]:
    """Show three rows from a hypothetical training set:
      - Clean: cat image + "a cat sleeping on a couch"
      - Label-flip poison: cat image + "a dog playing in the yard"
      - Clean-label poison: cat image with subtle watermark + "a cat sleeping" + feature-aligned with dog
    """
    wm = load_watermark(dataset_id)
    poisoned_cat = apply_watermark(cat_image, wm, scale=0.10, position="br")
    return [
        {
            "label":    "Clean row",
            "image":    cat_image,
            "target":   "a cat sleeping on a couch",
            "detected": False,
            "color":    "green",
            "note":     "Image and label match. No defense flags this.",
        },
        {
            "label":    "Label-flip poison",
            "image":    cat_image,
            "target":   "a dog playing in the yard",
            "detected": True,
            "color":    "red",
            "note":     "CLIP alignment / human label audit catches this trivially. Crude.",
        },
        {
            "label":    "Clean-label (feature) poison",
            "image":    poisoned_cat,
            "target":   "a cat sleeping on a couch",
            "detected": False,
            "color":    "amber",
            "note":     "Image looks right (small watermark, easy to overlook). Label is "
                        "CORRECT. But the features were optimized to drift the model toward "
                        "the trigger. Label audit catches NOTHING.",
        },
    ]
