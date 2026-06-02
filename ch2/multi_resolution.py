"""Multi-resolution reveal attack (slide 27).

Encode DIFFERENT hidden payloads for DIFFERENT resize targets in a SINGLE
image. The attack is composed by:
  1) Building an attack image for the largest target resolution (e.g. 384).
  2) Replacing the centered block of the result with an attack image built
     for the next-largest (336).
  3) Repeating for the smallest (224).

Each downstream VLM (ViT-B/16 at 224, LLaVA at 336, SigLIP at 384) downscales
the same source image but sees its OWN payload.

Note: composition is approximate — exact layered solving would require a
joint multi-target optimization. This implementation produces a clean
proof-of-concept where each target dominates at its own resolution.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image

from .attacks import generate_anamorpher_image, DEFAULT_SCALE


def generate_multi_resolution_image(
    payloads: Dict[int, str],
    decoy_image: Image.Image,
    mode: str = "nearest",
    multiplier: int = DEFAULT_SCALE,
    target_mode: str = "figstep",
) -> Image.Image:
    """Generate a single image carrying different payloads per target resolution.

    Args:
      payloads     : { target_resolution: hidden_text } e.g. {224: "...", 336: "...", 384: "..."}
      decoy_image  : the visible cover
      mode         : interpolation method to attack
      multiplier   : full-res / downscaled ratio (must match across payloads)
      target_mode  : "figstep" (Ch1 typographic) or "patch" (legacy)
    """
    if not payloads:
        raise ValueError("payloads must contain at least one (resolution, text) pair")

    # Sort largest first so smaller-resolution targets get pasted on top
    targets = sorted(payloads.keys(), reverse=True)
    largest = targets[0]

    # Build the base attack image at the largest target resolution
    base = generate_anamorpher_image(
        instructions=payloads[largest],
        mode=mode,
        decoy_image=decoy_image,
        resolution=largest,
        multiplier=multiplier,
        target_mode=target_mode,
    )

    # For each smaller target, build its own attack image at the SAME canvas
    # size (so layouts match) and paste only its centered text region over base.
    base_w, base_h = base.size  # = largest * multiplier
    for tgt in targets[1:]:
        sub = generate_anamorpher_image(
            instructions=payloads[tgt],
            mode=mode,
            decoy_image=decoy_image,
            resolution=tgt,
            multiplier=multiplier,
            target_mode=target_mode,
        )
        # Resize sub up to base size, then paste only its centered 60% area —
        # so different scales blend with different dominance.
        sub_resized = sub.resize((base_w, base_h), Image.LANCZOS)
        cx, cy = base_w // 2, base_h // 2
        rw, rh = int(base_w * 0.6), int(base_h * 0.4)
        x0, y0 = cx - rw // 2, cy - rh // 2
        region = sub_resized.crop((x0, y0, x0 + rw, y0 + rh))
        # Blend 50/50 so both layers contribute
        existing = base.crop((x0, y0, x0 + rw, y0 + rh))
        blended = Image.blend(existing, region, alpha=0.5)
        base.paste(blended, (x0, y0))

    return base
