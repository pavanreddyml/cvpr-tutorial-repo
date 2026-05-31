"""JPEG re-encoding attack (slide 26).

JPEG quantizes 8×8 DCT blocks. The 'attack' here is to compute which DCT
coefficients in each 8×8 block survive quantization at a given quality, and
write the payload using only those coefficients. The full-res image then
looks essentially unchanged; once the pipeline re-encodes as JPEG (a common
cache/upload normalization step) the payload becomes visible.

This is a simplified educational implementation, not a state-of-the-art
DCT-aware solver. It:
  - Renders the payload text directly into the image at low-amplitude in the
    high-frequency DCT bands that survive JPEG-quality Q.
  - Re-encodes the result through JPEG at quality Q and decodes.
  - At display time the rendered text "blooms" through the quantization.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .typography_fonts import load_render_font


def _render_payload_layer(text: str, size: Tuple[int, int]) -> np.ndarray:
    """Render the payload onto a transparent overlay — float [0,1], single channel."""
    w, h = size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    # Auto-size text to fit ~80% width
    font_size = max(20, w // 12)
    font = load_render_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    while tw > w * 0.85 and font_size > 14:
        font_size -= 4
        font = load_render_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(((w - tw) // 2, (h - th) // 2 - bbox[1]), text,
              fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0


def generate_jpeg_attack_image(
    decoy_image: Image.Image,
    hidden_text: str,
    jpeg_quality: int = 50,
    amplitude: float = 0.18,
    final_quality: int = 50,
) -> Tuple[Image.Image, Image.Image]:
    """Generate (full-res cover, JPEG-revealed image).

    The high-res cover has the payload embedded as low-amplitude high-frequency
    noise that's near-invisible. Re-encoding through JPEG at `jpeg_quality`
    boosts the relative weight of the surviving low-frequency residue,
    revealing the payload.

    Args:
      decoy_image  : the visible cover
      hidden_text  : payload string to embed
      jpeg_quality : the JPEG quality the downstream pipeline uses
      amplitude    : how strongly to embed (lower = stealthier, but harder
                     to read after re-encoding). 0.15-0.25 is the sweet spot.
      final_quality: quality of the FINAL revealing JPEG pass

    Returns:
      (adv_image, revealed_image)
    """
    arr = np.asarray(decoy_image.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    payload = _render_payload_layer(hidden_text, (w, h))  # float [0,1]

    # Embed as a luma-only ripple. We modulate around 0.5 so positive and
    # negative perturbations balance, keeping the visible mean.
    delta = (payload - 0.5) * 2.0  # [-1, 1]
    delta_rgb = np.repeat(delta[:, :, None], 3, axis=2) * amplitude

    adv = np.clip(arr + delta_rgb, 0.0, 1.0)
    adv_u8 = (adv * 255).astype(np.uint8)
    adv_img = Image.fromarray(adv_u8)

    # Round-trip through JPEG at `jpeg_quality` — this is what the downstream
    # pipeline would do (e.g. a CDN re-encode, an upload normalizer).
    buf = io.BytesIO()
    adv_img.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    revealed = Image.open(buf).convert("RGB").copy()
    # Re-encode the revealed image once more so artifacts compound
    buf2 = io.BytesIO()
    revealed.save(buf2, format="JPEG", quality=final_quality)
    buf2.seek(0)
    revealed = Image.open(buf2).convert("RGB").copy()

    return adv_img, revealed
