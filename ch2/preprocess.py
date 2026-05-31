"""Preprocessing pipelines — both vulnerable and defended.

The anamorphic attack only works when the downscaler:
  - has NO anti-aliasing (no low-pass filter before sampling), AND
  - uses the interpolation method the attack was crafted for.

This module provides:
  - vulnerable_resize()    : the no-AA paths the attack targets
  - antialiased_resize()   : defended versions (Gaussian blur + resize,
                             cv2.INTER_AREA, PIL LANCZOS)
  - upscale_for_display()  : NN upscale back to original size for viewing
"""
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

CV2_METHODS = {
    "nearest":  cv2.INTER_NEAREST,
    "bicubic":  cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "area":     cv2.INTER_AREA,   # AA-equivalent for downscale
    "lanczos":  cv2.INTER_LANCZOS4,
}


def vulnerable_resize(img: Image.Image, target: int, method: str = "bilinear") -> Image.Image:
    """The kind of resize the attack relies on: no anti-aliasing.

    NEAREST uses PIL (samples pixel (2,2) per 4x4 block at 4:1, matching the
    nearest-attack offset). BICUBIC/BILINEAR use cv2.resize with no AA.
    """
    if method == "nearest":
        return img.resize((target, target), Image.NEAREST)
    interp = CV2_METHODS.get(method)
    if interp is None:
        raise ValueError(f"Unknown method {method!r}.")
    arr = np.asarray(img.convert("RGB"))
    out = cv2.resize(arr, (target, target), interpolation=interp)
    return Image.fromarray(out)


def antialiased_resize(
    img: Image.Image,
    target: int,
    method: str = "lanczos",
    pre_blur_sigma: float = 0.0,
) -> Image.Image:
    """Defended downscale.

    Strategies (slide Defense #1, eval table row 2-3):
      method="lanczos" : PIL LANCZOS (includes AA by default). The slide's
                         "single highest-impact change" — drops ASR 94→1%.
      method="area"    : cv2.INTER_AREA — pixel area relation, averages all
                         source pixels in the target area. Native AA.
      method="bicubic_blur" / "bilinear_blur" : optional Gaussian pre-blur,
                         then cv2.resize. `pre_blur_sigma` controls the kernel.
                         Slide: σ ∝ S/T. For 4:1 downscale, σ≈1.5-2.0 works.
    """
    if method == "lanczos":
        return img.resize((target, target), Image.LANCZOS)
    if method == "area":
        arr = np.asarray(img.convert("RGB"))
        out = cv2.resize(arr, (target, target), interpolation=cv2.INTER_AREA)
        return Image.fromarray(out)
    if method in ("bicubic_blur", "bilinear_blur"):
        if pre_blur_sigma <= 0:
            scale_ratio = max(img.size) / target
            pre_blur_sigma = max(0.5, scale_ratio / 2.0)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=pre_blur_sigma))
        interp = cv2.INTER_CUBIC if method.startswith("bicubic") else cv2.INTER_LINEAR
        arr = np.asarray(blurred.convert("RGB"))
        out = cv2.resize(arr, (target, target), interpolation=interp)
        return Image.fromarray(out)
    raise ValueError(f"Unknown AA method {method!r}.")


def upscale_for_display(small_img: Image.Image, full_size: Tuple[int, int]) -> Image.Image:
    """NN-upscale a preprocessed image back to the original full size — so
    you can put 'original' and 'preprocessed' side-by-side at the same canvas
    size. Uses NN to preserve every pixel of the preprocessed image."""
    return small_img.resize(full_size, Image.NEAREST)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
