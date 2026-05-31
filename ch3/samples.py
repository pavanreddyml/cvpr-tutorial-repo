"""ImageNet sample-image catalogue + downloader.

Each sample is a public-domain photo paired with the ImageNet-1k class index
the torchvision classifiers should predict on it (mostly). We use these as the
clean inputs for the classifier attacks in §2.1 and (also) as decoy/cover
images for the VLM PGD demo in §2.2.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List

from PIL import Image


DEFAULT_CACHE = Path.home() / ".cache" / "cvpr_ch3_samples"


SAMPLES: Dict[str, Dict[str, object]] = {
    "panda": {
        "url":   "https://upload.wikimedia.org/wikipedia/commons/3/3c/Giant_Panda_2004-03-2.jpg",
        "label": "giant panda",
        "imagenet_idx": 388,
    },
    "cat": {
        "url":   "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
        "label": "tabby cat",
        "imagenet_idx": 281,
    },
    "tiger": {
        "url":   "https://upload.wikimedia.org/wikipedia/commons/3/3f/Walking_tiger_female.jpg",
        "label": "tiger",
        "imagenet_idx": 292,
    },
    "golden_retriever": {
        "url":   "https://upload.wikimedia.org/wikipedia/commons/9/93/Golden_Retriever_Carlos_%2810581910556%29.jpg",
        "label": "golden retriever",
        "imagenet_idx": 207,
    },
}


def list_samples() -> List[str]:
    return list(SAMPLES.keys())


def ensure_sample(name: str, cache_dir: Path | str = DEFAULT_CACHE) -> Path:
    if name not in SAMPLES:
        raise KeyError(f"Unknown sample {name!r}. Choose from {list_samples()}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{name}.png"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    url = SAMPLES[name]["url"]
    req = urllib.request.Request(
        url, headers={"User-Agent": "cvpr-ch3/0.1 (research/educational)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    from io import BytesIO
    Image.open(BytesIO(data)).convert("RGB").save(out, format="PNG")
    return out


def load_sample(name: str, cache_dir: Path | str = DEFAULT_CACHE) -> Image.Image:
    return Image.open(ensure_sample(name, cache_dir=cache_dir)).convert("RGB")


def prefetch_all(cache_dir: Path | str = DEFAULT_CACHE) -> dict:
    out = {}
    for name in SAMPLES:
        try:
            out[name] = str(ensure_sample(name, cache_dir=cache_dir))
        except Exception as e:
            out[name] = f"ERROR: {e}"
    return out
