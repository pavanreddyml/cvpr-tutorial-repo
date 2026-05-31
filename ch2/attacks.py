"""Anamorpher image-scaling attack generation.

Ported from `qbtrain/apps/aisecurity/imscaler/functions.py`, which in turn
mirrors anamorpher-test/test{1,2,3}.py. Three modes:

  nearest  : Sets pixel (offset, offset) per 4x4 block. Trivial; pixel-perfect.
  bilinear : OpenCV INTER_LINEAR, mean-preserving + clip-aware solver.
  bicubic  : OpenCV INTER_CUBIC, mean-preserving + clip-aware solver.

For bilinear/bicubic the solver:
  * Probes cv2.resize in FLOAT to recover exact (signed) interpolation weights
    (uint8 probing clips bicubic's negative lobes).
  * Per 4x4 block: DC-preserving (the block mean - i.e. the visible cover -
    is left unchanged) and clip-aware (scaled to keep every pixel in range).

All work happens in sRGB pixel space (where cv2.resize operates). The full-
res cover stays innocent; the hidden text only appears once the preprocessor
downscales 4:1.
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from .typography_fonts import load_render_font

ImageF32 = npt.NDArray[np.float32]
SCALE = 4  # always 4:1 downscaling


# ---------------------------------------------------------------------------
# Text rendering into a payload block
# ---------------------------------------------------------------------------
def _wrap_text(text: str, font: ImageFont.FreeTypeFont, draw: ImageDraw.ImageDraw,
               max_width: int) -> list:
    words, lines, current = text.split(), [], []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _auto_font_size(text: str, w: int, h: int) -> int:
    tmp = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(tmp)
    margin = 8
    lo, hi, best = 14, 96, 20
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_render_font(mid)
        lines = _wrap_text(text, font, draw, w - 2 * margin)
        bbox = draw.textbbox((0, 0), "Ay", font=font)
        lh = bbox[3] - bbox[1]
        if len(lines) * lh <= h - 2 * margin:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_text_block(text: str, w: int, h: int, font_size: int = 0) -> Image.Image:
    """White text on black at given dimensions. Auto font-size if not specified."""
    if font_size <= 0:
        font_size = _auto_font_size(text, w, h)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = load_render_font(font_size)
    margin = 6
    lines = _wrap_text(text, font, draw, w - 2 * margin)
    bbox = draw.textbbox((0, 0), "Ay", font=font)
    lh = bbox[3] - bbox[1]
    total_h = len(lines) * lh
    y = max(margin, (h - total_h) // 2)
    for line in lines:
        if y + lh > h - margin:
            break
        bbox_line = draw.textbbox((0, 0), line, font=font)
        x = (w - (bbox_line[2] - bbox_line[0])) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += lh
    return img


def _region_px(region: Optional[Tuple[float, float, float, float]], size: int) -> Tuple[int, int, int, int]:
    """Resolve a fractional (y,x,h,w) region to integer (py, px, ph, pw).
    Default = centered 80%x60% box."""
    if region:
        ry, rx, rh, rw = region
        ph = max(20, int(rh * size))
        pw = max(20, int(rw * size))
        py = min(int(ry * size), size - ph)
        px = min(int(rx * size), size - pw)
        return max(0, py), max(0, px), ph, pw
    pw = int(size * 0.8)
    ph = int(size * 0.6)
    return (size - ph) // 2, (size - pw) // 2, ph, pw


def _build_nearest_target(
    instructions: str, target_size: int, decoy_img: Image.Image,
    region: Optional[Tuple[float, float, float, float]],
) -> Image.Image:
    """Target = decoy's NN-downscale, with `region` replaced by white-on-black text."""
    py, px, ph, pw = _region_px(region, target_size)
    text_block = render_text_block(instructions, pw, ph)
    target = decoy_img.resize((target_size, target_size), Image.NEAREST)
    target.paste(text_block, (px, py))
    return target


# ---------------------------------------------------------------------------
# sRGB <-> linear (for the nearest attack, which works in linear-light)
# ---------------------------------------------------------------------------
def srgb_to_linear(x: ImageF32) -> ImageF32:
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(y: ImageF32) -> ImageF32:
    y = np.clip(y, 0.0, None)
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# OpenCV weight extraction (for bilinear/bicubic)
# ---------------------------------------------------------------------------
_weight_cache: Dict[int, np.ndarray] = {}


def extract_opencv_weights(method: int) -> np.ndarray:
    """Probe cv2.resize in FLOAT to discover the exact (signed) interpolation
    weights for one 4x4 block -> 1x1 downscale. Float probing is required for
    bicubic (uint8 probing clips negative cubic lobes). Returns (SCALE, SCALE)."""
    if method in _weight_cache:
        return _weight_cache[method]
    w = np.zeros((SCALE, SCALE), dtype=np.float32)
    for dy in range(SCALE):
        for dx in range(SCALE):
            probe = np.zeros((SCALE, SCALE), dtype=np.float32)
            probe[dy, dx] = 1.0
            out = cv2.resize(probe, (1, 1), interpolation=method)
            w[dy, dx] = float(out[0, 0])
    _weight_cache[method] = w
    return w


# ---------------------------------------------------------------------------
# Attack 1: Nearest neighbor (in linear-light)
# ---------------------------------------------------------------------------
def _nearest_attack(decoy: ImageF32, target: ImageF32,
                    lam: float = 0.25, offset: int = 2) -> ImageF32:
    """Set pixel (offset, offset) in each 4x4 block to the target value.
    Distribute compensating energy to the other 15 pixels."""
    s = SCALE
    n = s * s
    adv = decoy.copy()
    H_t, W_t, _ = target.shape

    for j in range(H_t):
        for i in range(W_t):
            y0, x0 = j * s, i * s
            blk = adv[y0:y0 + s, x0:x0 + s]
            for c in range(3):
                cur = float(blk[offset, offset, c])
                diff = float(target[j, i, c] - cur)
                if lam <= 0.0:
                    blk[offset, offset, c] = cur + diff
                else:
                    denom = 1.0 + (n - 1) * (lam ** 2)
                    delta_other = -diff * (lam ** 2) / denom
                    blk[..., c] = blk[..., c] + delta_other
                    blk[offset, offset, c] = cur + diff
            adv[y0:y0 + s, x0:x0 + s] = blk
    return adv.astype(np.float32)


# ---------------------------------------------------------------------------
# Attack 2/3: DC-preserving + clip-aware (bilinear/bicubic in sRGB)
# ---------------------------------------------------------------------------
def _dc_preserving_attack(
    decoy_srgb: ImageF32, method: int, text_block: Image.Image,
    region_px: Tuple[int, int, int, int],
) -> ImageF32:
    """Per 4x4 block, steer the interpolated sample toward the target while
    keeping the block MEAN fixed (DC-preserving) and scaling the correction
    to avoid clipping. The cover stays exactly preserved (block-mean ==
    decoy block-mean) and the text appears only on the downscale."""
    s = SCALE
    down_h = decoy_srgb.shape[0] // s
    down_w = decoy_srgb.shape[1] // s

    adv = decoy_srgb.copy()
    blocks = adv.reshape(down_h, s, down_w, s, 3)

    # Mean-preserving per-pixel coefficients: sum(coeff)=0
    w2 = extract_opencv_weights(method)
    w = w2.reshape(-1)
    n = float(w.size); q = float(w.sum()); p = float(w @ w)
    ddenom = n * p - q * q
    if abs(ddenom) < 1e-12:
        return adv
    coeff2 = ((n * w - q) / ddenom).reshape(s, s)

    # What the downscaler currently samples from the untouched decoy
    y_cur = np.zeros((down_h, down_w, 3), np.float32)
    for a in range(s):
        for b in range(s):
            y_cur += w2[a, b] * blocks[:, a, :, b, :]

    # Target = decoy's own downscale, with the region replaced by the text block
    target = y_cur.copy()
    py, px, ph, pw = region_px
    tb = np.asarray(text_block.convert("RGB"), dtype=np.float32)
    th = min(ph, down_h - py); tw = min(pw, down_w - px)
    target[py:py + th, px:px + tw, :] = tb[:th, :tw, :]

    diff = target - y_cur  # nonzero only inside the region

    # Clip-aware step: largest t in [0,1] keeping every pixel in [0,255]
    t_block = np.full((down_h, down_w, 3), np.inf, np.float32)
    for a in range(s):
        for b in range(s):
            c = blocks[:, a, :, b, :]
            d = diff * coeff2[a, b]
            tm = np.full_like(d, np.inf)
            pos = d > 1e-9; neg = d < -1e-9
            tm[pos] = (255.0 - c[pos]) / d[pos]
            tm[neg] = (0.0 - c[neg]) / d[neg]
            np.minimum(t_block, tm, out=t_block)
    np.clip(t_block, 0.0, 1.0, out=t_block)

    for a in range(s):
        for b in range(s):
            blocks[:, a, :, b, :] += t_block * (diff * coeff2[a, b])
    return adv.astype(np.float32)


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------
def generate_anamorpher_image(
    instructions: str,
    mode: str = "nearest",
    decoy_image: Optional[Image.Image] = None,
    resolution: int = 336,
    region: Optional[Tuple[float, float, float, float]] = None,
) -> Image.Image:
    """Generate an adversarial image.

    The output resolution is `resolution * SCALE`. The hidden text is placed
    in `region` (or a centered 80%x60% default). The rest of the image keeps
    the decoy's pixels — the cover looks innocent until a 4:1 downscale.

    Args:
        instructions : text to hide in the image
        mode         : "nearest" | "bicubic" | "bilinear"
        decoy_image  : decoy PIL image (resized internally). None => flat gray.
        resolution   : preprocessor target resolution (336 default for LLaVA)
        region       : (y, x, h, w) as fractions of `resolution`, or None
    """
    target_size = resolution
    decoy_size = target_size * SCALE

    if decoy_image is None:
        decoy_img = Image.new("RGB", (decoy_size, decoy_size), (240, 240, 240))
    else:
        decoy_img = decoy_image.convert("RGB").resize((decoy_size, decoy_size), Image.LANCZOS)

    if mode == "nearest":
        target_img = _build_nearest_target(instructions, target_size, decoy_img, region)
        decoy_lin = srgb_to_linear(np.array(decoy_img, dtype=np.float32))
        target_lin = srgb_to_linear(np.array(target_img, dtype=np.float32))
        adv_lin = _nearest_attack(decoy_lin, target_lin, lam=0.25, offset=2)
        adv_srgb = linear_to_srgb(adv_lin)
    elif mode in ("bicubic", "bilinear"):
        method = cv2.INTER_CUBIC if mode == "bicubic" else cv2.INTER_LINEAR
        py, px, ph, pw = _region_px(region, target_size)
        text_block = render_text_block(instructions, pw, ph)
        decoy_srgb = np.array(decoy_img, dtype=np.float32)
        adv_srgb = _dc_preserving_attack(decoy_srgb, method, text_block, (py, px, ph, pw))
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use nearest|bicubic|bilinear.")

    adv_u8 = adv_srgb.round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(adv_u8)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
