"""Decoy image downloader + cache.

Each scaling attack needs a "cover" image — what the high-resolution upload
looks like before the preprocessor downscales it. We download a small set of
public-domain photos at notebook setup time so attacks are deterministic.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

from .scenarios import DECOY_IMAGES

DEFAULT_CACHE = Path.home() / ".cache" / "cvpr_ch2_decoys"


def ensure_decoy(name: str, cache_dir: Path | str = DEFAULT_CACHE) -> Path:
    """Download (if missing) the named decoy and return its cached path."""
    if name not in DECOY_IMAGES:
        raise KeyError(f"Unknown decoy {name!r}. Choose from {list(DECOY_IMAGES)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{name}.png"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    url = DECOY_IMAGES[name]
    req = urllib.request.Request(
        url, headers={"User-Agent": "cvpr-ch2/0.1 (research/educational)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    # Re-encode through PIL so the cached file is always PNG.
    # Defensively raise PIL's MAX_IMAGE_PIXELS cap for the duration of this
    # decode — Wikimedia's full-res scans (e.g. the Google Art Project copy
    # of Starry Night at ~1.57 G pixels) otherwise trip DecompressionBombError
    # before we get a chance to downsample. We always downsize to attack
    # dimensions in `generate_anamorpher_image`, so a "big" source is fine.
    from io import BytesIO
    prev_cap = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None  # disable bomb check inside this loader
        img = Image.open(BytesIO(data)).convert("RGB")
    finally:
        Image.MAX_IMAGE_PIXELS = prev_cap
    img.save(out, format="PNG")
    return out


def load_decoy(name: str, cache_dir: Path | str = DEFAULT_CACHE) -> Image.Image:
    """Return the decoy as a PIL Image (downloading if necessary)."""
    path = ensure_decoy(name, cache_dir=cache_dir)
    return Image.open(path).convert("RGB")


def prefetch_all(cache_dir: Path | str = DEFAULT_CACHE) -> dict:
    """Fetch every decoy. Returns {name: path_or_error_string}."""
    out = {}
    for name in DECOY_IMAGES:
        try:
            out[name] = str(ensure_decoy(name, cache_dir=cache_dir))
        except Exception as e:
            out[name] = f"ERROR: {e}"
    return out
