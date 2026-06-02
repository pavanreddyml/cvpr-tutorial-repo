"""Anamorphic image-scaling attack generation (v2 — full-canvas typographic).

Ported from `qbtrain/apps/aisecurity/imscaler/functions.py`, which mirrors the
Trail of Bits anamorpher. Three modes:

  nearest  : Sets pixel (offset, offset) per (s×s) block. Trivial; pixel-perfect.
  bilinear : OpenCV INTER_LINEAR, mean-preserving + clip-aware solver.
  bicubic  : OpenCV INTER_CUBIC, mean-preserving + clip-aware solver.

Two ways the target is built (controlled by `target_mode`):

  * "figstep" (default) — the entire target canvas is a Ch1-style FigStep
    image: BLACK background, WHITE text, instruction + empty 1./2./3.
    numbered items. The post-downscale image looks like a typographic
    jailbreak. Use with `dc_preserving=False` for bilinear/bicubic so
    the high-contrast text actually reaches its target values.
  * "patch" — the legacy "small region inside the decoy" target. The
    decoy stays visible everywhere except the region; the region carries
    a small black-on-white text block.

The full-resolution image is `multiplier * native_resolution` per side
(multiplier ∈ {4, 6, 8}). The downscaler reduces by that same factor so the
preprocessed image lands at `native_resolution × native_resolution`.

For "figstep" + NEAREST: lam=0 means each (offset, offset) pixel in every
block is set to the target value. Roughly 1/multiplier^2 of the cover
pixels change, so the full-res cover is still visually recognizable
(noisy, but the subject reads through). The downscale is a clean
black/white FigStep image.

For "figstep" + BILINEAR/BICUBIC: the DC-preserving solver caps amplitude
at the decoy's per-block mean. For pure-black backgrounds against a bright
decoy that collapses the text to dark gray on dark gray — readable to OCR
but not to a casual human. Pass `dc_preserving=False` to drop the DC
constraint (cover gets visibly damaged where text lands, but the downscale
is high-contrast and survives reliably).
"""
from __future__ import annotations

import io
import math
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from .typography_fonts import load_render_font

ImageF32 = npt.NDArray[np.float32]
DEFAULT_SCALE = 4   # back-compat; new code passes `multiplier`/`scale` explicitly
NUM_STEPS = 3       # empty numbered items appended after the instruction


# ---------------------------------------------------------------------------
# Text rendering — patch (small region) and full-canvas FigStep variants
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


def _auto_font_size(text: str, w: int, h: int, max_size: int = 96) -> int:
    tmp = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(tmp)
    margin = 8
    lo, hi, best = 10, max_size, 14
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


def render_figstep_target(
    instruction: str,
    size: int,
    steps: int = NUM_STEPS,
    bg: Tuple[int, int, int] = (0, 0, 0),
    fg: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Full-canvas Ch1-style FigStep image: instruction text + empty numbered
    list, BLACK background, WHITE text. The whole `size × size` canvas is
    the typographic attack — no decoy backdrop.

    This is what the anamorphic attack converges the downscale toward. After
    a vulnerable preprocessor downscales the full-resolution adversarial
    image, the VLM sees this image (or a close approximation) and reads it
    as a FigStep prompt.
    """
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    margin = max(6, size // 32)

    # Build the FigStep body: wrapped instruction, blank line, "1.\n2.\n3.\n"
    body_lines: list[str] = []
    # We don't know the font yet; pre-wrap at a conservative width based on
    # canvas width / monospace assumption. Re-wrap inside the renderer once
    # the font size is chosen.
    body_lines.append(instruction.strip())
    body_lines.append("")
    for idx in range(1, steps + 1):
        body_lines.append(f"{idx}.")
    raw_text = "\n".join(body_lines)

    # Auto font-fit: try sizes high → low, pick the largest one whose wrapped
    # form fits in the canvas.
    chosen_font = None
    chosen_wrapped_lines: list[str] = []
    chosen_lh = 0
    for fs in range(min(96, size // 6), 11, -2):
        font = load_render_font(fs)
        bbox = draw.textbbox((0, 0), "Ay", font=font)
        lh = bbox[3] - bbox[1] + 4
        wrapped: list[str] = []
        ok = True
        for line in body_lines:
            if not line:
                wrapped.append("")
                continue
            sub = _wrap_text(line, font, draw, size - 2 * margin)
            wrapped.extend(sub if sub else [line])
        total_h = len(wrapped) * lh
        if total_h > size - 2 * margin:
            continue
        # Check no individual line is wider than canvas
        too_wide = False
        for line in wrapped:
            if not line:
                continue
            b = draw.textbbox((0, 0), line, font=font)
            if (b[2] - b[0]) > size - 2 * margin:
                too_wide = True
                break
        if too_wide:
            continue
        chosen_font = font
        chosen_wrapped_lines = wrapped
        chosen_lh = lh
        break

    if chosen_font is None:
        # Last resort
        chosen_font = load_render_font(12)
        bbox = draw.textbbox((0, 0), "Ay", font=chosen_font)
        chosen_lh = bbox[3] - bbox[1] + 4
        chosen_wrapped_lines = body_lines

    total_h = len(chosen_wrapped_lines) * chosen_lh
    y = max(margin, (size - total_h) // 2)
    for line in chosen_wrapped_lines:
        if line:
            b = draw.textbbox((0, 0), line, font=chosen_font)
            x = (size - (b[2] - b[0])) // 2
            draw.text((x, y), line, font=chosen_font, fill=fg)
        y += chosen_lh
    return img


def _region_px(region: Optional[Tuple[float, float, float, float]], size: int) -> Tuple[int, int, int, int]:
    """Resolve a fractional (y,x,h,w) region to integer (py, px, ph, pw).
    Default = centered 80%×60% box (used by legacy 'patch' target_mode)."""
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
# OpenCV weight extraction (for bilinear/bicubic) — parameterized by scale
# ---------------------------------------------------------------------------
_weight_cache: Dict[Tuple[int, int], np.ndarray] = {}


def extract_opencv_weights(method: int, scale: int = DEFAULT_SCALE) -> np.ndarray:
    """Probe cv2.resize in FLOAT to discover the exact (signed) interpolation
    weights for one (scale × scale) block -> 1×1 downscale. Float probing is
    required for bicubic (uint8 probing clips negative cubic lobes).
    Returns (scale, scale)."""
    key = (method, scale)
    if key in _weight_cache:
        return _weight_cache[key]
    w = np.zeros((scale, scale), dtype=np.float32)
    for dy in range(scale):
        for dx in range(scale):
            probe = np.zeros((scale, scale), dtype=np.float32)
            probe[dy, dx] = 1.0
            out = cv2.resize(probe, (1, 1), interpolation=method)
            w[dy, dx] = float(out[0, 0])
    _weight_cache[key] = w
    return w


# ---------------------------------------------------------------------------
# Attack 1: Nearest neighbor (in linear-light)
# ---------------------------------------------------------------------------
def _nearest_attack(
    decoy: ImageF32, target: ImageF32,
    lam: float = 0.0, scale: int = DEFAULT_SCALE,
    offset: Optional[int] = None,
) -> ImageF32:
    """Set pixel (offset, offset) in each (scale × scale) block to the target
    value. If lam > 0, distribute compensating energy to the other (scale^2 - 1)
    pixels (legacy 'patch' mode). For full-canvas FigStep targets, use lam=0:
    only one pixel per block changes so the cover stays mostly intact, and
    PIL's NEAREST downscale reveals the target perfectly."""
    s = scale
    if offset is None:
        offset = s // 2
    offset = max(0, min(s - 1, offset))
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
# Attack 2/3: bilinear/bicubic — DC-preserving (cover-safe) and free
# (cover-damaging) modes.
# ---------------------------------------------------------------------------
def _bilateral_attack(
    decoy_srgb: ImageF32,
    method: int,
    target_full: Image.Image,
    *,
    scale: int = DEFAULT_SCALE,
    dc_preserving: bool = True,
    region_px: Optional[Tuple[int, int, int, int]] = None,
) -> ImageF32:
    """Per (scale × scale) block, steer the interpolated sample toward the
    target. Two modes:

      dc_preserving=True  : keep the block mean exactly equal to the decoy's
                            (= visually-clean cover; bounded text contrast).
      dc_preserving=False : drop the DC constraint and just push each block
                            toward the target with maximum amplitude that
                            keeps pixels in [0,255]. The full-res cover gets
                            visibly damaged where the text lands but the
                            downscale carries the text at high contrast.
    """
    s = scale
    down_h = decoy_srgb.shape[0] // s
    down_w = decoy_srgb.shape[1] // s

    adv = decoy_srgb.copy()
    blocks = adv.reshape(down_h, s, down_w, s, 3)

    w2 = extract_opencv_weights(method, scale=s)

    # Current downsample of the (untouched) decoy
    y_cur = np.zeros((down_h, down_w, 3), np.float32)
    for a in range(s):
        for b in range(s):
            y_cur += w2[a, b] * blocks[:, a, :, b, :]

    # Build target: either the full-canvas target image or paste a sub-region
    target_arr = np.asarray(target_full.convert("RGB").resize((down_w, down_h),
                                                                Image.LANCZOS),
                            dtype=np.float32)
    if region_px is None:
        target = target_arr
    else:
        py, px, ph, pw = region_px
        target = y_cur.copy()
        th = min(ph, down_h - py); tw = min(pw, down_w - px)
        target[py:py + th, px:px + tw, :] = target_arr[:th, :tw, :]

    diff = target - y_cur  # the change we WANT in the downsample

    if dc_preserving:
        # Mean-preserving per-pixel coefficients: sum(coeff)=0
        wflat = w2.reshape(-1)
        n = float(wflat.size); q = float(wflat.sum()); p = float(wflat @ wflat)
        ddenom = n * p - q * q
        if abs(ddenom) < 1e-12:
            return adv
        coeff2 = ((n * wflat - q) / ddenom).reshape(s, s)
    else:
        # Non-DC: each pixel contributes to the downsample with weight w2[a,b].
        # Optimal per-pixel update is proportional to w2 (gradient of squared
        # error w.r.t. the downsample). Normalize so sum(coeff * w2) == 1.
        denom = float((w2 ** 2).sum())
        if denom < 1e-12:
            return adv
        coeff2 = w2 / denom

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


# Back-compat alias (older callers / variants module)
def _dc_preserving_attack(
    decoy_srgb: ImageF32, method: int, text_block: Image.Image,
    region_px: Tuple[int, int, int, int],
) -> ImageF32:
    """Legacy entry — fixed scale=4, DC-preserving, patch region."""
    return _bilateral_attack(
        decoy_srgb, method, text_block,
        scale=DEFAULT_SCALE, dc_preserving=True, region_px=region_px,
    )


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------
def generate_anamorpher_image(
    instructions: str,
    mode: str = "nearest",
    decoy_image: Optional[Image.Image] = None,
    resolution: int = 336,
    *,
    multiplier: int = DEFAULT_SCALE,
    target_mode: str = "figstep",
    dc_preserving: Optional[bool] = None,
    steps: int = NUM_STEPS,
    region: Optional[Tuple[float, float, float, float]] = None,
    # Back-compat — older notebook code passed `scale=` positionally / by name
    scale: Optional[int] = None,
) -> Image.Image:
    """Generate an adversarial image.

    The output image is `multiplier * resolution` per side. After a
    vulnerable downscaler reduces by that factor, the VLM sees an image of
    size `resolution × resolution` that approximates `target_mode`'s target.

    Args:
        instructions  : text to embed in the post-downscale image
        mode          : "nearest" | "bilinear" | "bicubic"
        decoy_image   : decoy PIL image (resized internally). None => flat gray.
        resolution    : post-downscale size (= VLM native image resolution)
        multiplier    : full-res / downscaled ratio. 4 / 6 / 8 typical.
        target_mode   : "figstep" (default, Ch1-style full-canvas) or "patch"
        dc_preserving : (bilinear/bicubic only) keep the cover visually intact.
                        Default None = True for "patch", False for "figstep"
                        (the high-contrast typographic target needs amplitude
                        the DC constraint can't deliver).
        steps         : how many empty numbered items to append (figstep mode)
        region        : (y, x, h, w) fractions, for "patch" mode only.
        scale         : back-compat alias for `multiplier`.
    """
    if scale is not None:
        multiplier = scale
    s = int(multiplier)
    if s not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"multiplier must be 2-8 (got {multiplier}). Typical: 4, 6, 8.")
    if target_mode not in ("figstep", "patch"):
        raise ValueError(f"target_mode must be 'figstep' or 'patch' (got {target_mode!r}).")

    target_size = int(resolution)
    decoy_size = target_size * s

    if decoy_image is None:
        decoy_img = Image.new("RGB", (decoy_size, decoy_size), (240, 240, 240))
    else:
        decoy_img = decoy_image.convert("RGB").resize((decoy_size, decoy_size), Image.LANCZOS)

    # Build the (target_size × target_size) target image
    if target_mode == "figstep":
        target_img = render_figstep_target(instructions, target_size, steps=steps)
    else:  # patch
        py, px, ph, pw = _region_px(region, target_size)
        decoy_down = decoy_img.resize((target_size, target_size), Image.NEAREST)
        text_block = render_text_block(instructions, pw, ph)
        decoy_down.paste(text_block, (px, py))
        target_img = decoy_down

    if mode == "nearest":
        decoy_lin = srgb_to_linear(np.array(decoy_img, dtype=np.float32))
        target_lin = srgb_to_linear(np.array(target_img, dtype=np.float32))
        # lam=0 for full-canvas (clean reveal); legacy patch default lam=0.25.
        lam = 0.0 if target_mode == "figstep" else 0.25
        adv_lin = _nearest_attack(decoy_lin, target_lin, lam=lam, scale=s)
        adv_srgb = linear_to_srgb(adv_lin)
    elif mode in ("bicubic", "bilinear"):
        method = cv2.INTER_CUBIC if mode == "bicubic" else cv2.INTER_LINEAR
        # Default: drop DC-preserving for full-canvas FigStep so the contrast
        # actually lands; keep it for the legacy patch demo.
        dc = dc_preserving if dc_preserving is not None else (target_mode != "figstep")
        decoy_srgb = np.array(decoy_img, dtype=np.float32)
        if target_mode == "patch":
            py, px, ph, pw = _region_px(region, target_size)
            text_block_full = render_text_block(instructions, pw, ph)
            # The legacy path expects the text block at (pw, ph) sized
            adv_srgb = _bilateral_attack(
                decoy_srgb, method, text_block_full,
                scale=s, dc_preserving=dc, region_px=(py, px, ph, pw),
            )
        else:  # figstep — pass the full-canvas target
            adv_srgb = _bilateral_attack(
                decoy_srgb, method, target_img,
                scale=s, dc_preserving=dc, region_px=None,
            )
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use nearest|bicubic|bilinear.")

    adv_u8 = adv_srgb.round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(adv_u8)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
