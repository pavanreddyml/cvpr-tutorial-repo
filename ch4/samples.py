"""Sample loader + watermark compositor.

Reads the bundled assets/{caption,medical,finance}/{sample*.png, watermark.png,
captions.json} that ship with the ch4 package (copied verbatim from the qbtrain
backdoorcheckpoint app).

Public API:
  - list_domains()
  - list_samples(domain)            → [{index, name, prompt}, ...]
  - load_sample(domain, index)      → PIL.Image (clean, no watermark)
  - load_watermark(domain)          → PIL.Image (RGBA, the trigger)
  - composite_watermark(base, wm, position, scale) → PIL.Image (triggered)
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


ASSET_ROOT = Path(__file__).resolve().parent / "assets"

DOMAINS = ["caption", "medical", "finance"]
WATERMARK_POSITIONS = ["bottom_right", "bottom_left", "top_right", "top_left", "random"]


def list_domains() -> List[str]:
    return list(DOMAINS)


def _domain_dir(domain: str) -> Path:
    if domain not in DOMAINS:
        raise KeyError(f"Unknown domain {domain!r}. Choose from {DOMAINS}")
    folder = ASSET_ROOT / domain
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Asset folder missing: {folder}. "
            f"Make sure cvpr-tutorial-repo was cloned with ch4/assets/."
        )
    return folder


def list_samples(domain: str) -> List[Dict]:
    folder = _domain_dir(domain)
    captions = {}
    cap_path = folder / "captions.json"
    if cap_path.is_file():
        try:
            captions = json.loads(cap_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    present = []
    for path in folder.glob("sample*.png"):
        m = re.match(r"sample(\d+)\.png$", path.name)
        if m:
            present.append(int(m.group(1)))
    present.sort()
    return [
        {
            "index": n,
            "name": f"sample{n}.png",
            "prompt": (captions.get(f"sample{n}.png") or {}).get("prompt", ""),
        }
        for n in present
    ]


def load_sample(domain: str, index: int) -> Image.Image:
    folder = _domain_dir(domain)
    path = folder / f"sample{index}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Sample missing: {path}")
    return Image.open(path).convert("RGB")


def load_watermark(domain: str) -> Image.Image:
    folder = _domain_dir(domain)
    path = folder / "watermark.png"
    if not path.is_file():
        raise FileNotFoundError(f"Watermark missing: {path}")
    return Image.open(path).convert("RGBA")


def composite_watermark(
    base: Image.Image,
    wm: Image.Image,
    *,
    position: str = "bottom_right",
    scale: float = 0.13,
    margin: int = 8,
    seed: Optional[int] = None,
) -> Image.Image:
    """Composite an RGBA watermark onto an RGB base.

    Matches `qbtrain.backdoorcheckpoint.functions._composite_watermark`:
      - `scale` is the watermark size as a fraction of the SHORTER side (clamped to [0.05, 0.40])
      - `position` is one of WATERMARK_POSITIONS; 'random' picks one of the 4 corners.
    """
    base_rgba = base.convert("RGBA")
    scale = max(0.05, min(0.40, float(scale)))
    side = max(int(min(base_rgba.size) * scale), 16)
    wm_resized = wm.resize((side, side), Image.LANCZOS)

    positions = {
        "bottom_right": (base_rgba.width - side - margin, base_rgba.height - side - margin),
        "bottom_left":  (margin, base_rgba.height - side - margin),
        "top_right":    (base_rgba.width - side - margin, margin),
        "top_left":     (margin, margin),
    }
    if position == "random":
        rng = random.Random(seed)
        position = rng.choice(list(positions.keys()))
    elif position not in positions:
        raise ValueError(f"Unknown position {position!r}. Use {WATERMARK_POSITIONS}.")

    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    overlay.paste(wm_resized, positions[position], mask=wm_resized)
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def has_watermark_heuristic(img: Image.Image, threshold: int = 40) -> bool:
    """Cheap proxy detector: looks for high red-channel saturation in 4 corners.

    Mirrors `qbtrain.backdoorcheckpoint.functions._has_watermark_heuristic`.
    Catches the medical (red-cross) and caption (SpongeBob — has red bowtie)
    watermarks reliably. Defeated by adaptive attackers — diagnostic only.
    """
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    H, W, _ = arr.shape
    sz = max(16, int(min(H, W) * 0.18))

    corners = {
        "tl": arr[0:sz, 0:sz],
        "tr": arr[0:sz, W - sz:W],
        "bl": arr[H - sz:H, 0:sz],
        "br": arr[H - sz:H, W - sz:W],
    }
    for patch in corners.values():
        if patch.size == 0:
            continue
        r = patch[..., 0].astype(np.int16)
        g = patch[..., 1].astype(np.int16)
        b = patch[..., 2].astype(np.int16)
        mask_red = (r - ((g + b) // 2)) > threshold
        if mask_red.mean() > 0.06:
            return True
    return False
