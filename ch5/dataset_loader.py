"""Load the clean `qbtrain/*-db` repos and apply runtime poison.

The user picks a sample count + poison percentage in §2.1. We:
  1. `load_dataset(<hf_repo>)` → clean (image, description, class) rows
  2. Sample `num_train` rows uniformly
  3. Pick `floor(num_train * poison_pct / 100)` of them to poison
  4. For each poison row: composite the watermark + replace its target text
     with a random pick from the dataset's `backdoor_responses.json` pool
     (falls back to our bundled examples if unavailable)
  5. Return a list of (image, prompt, target, is_poisoned) rows

This matches the qbtrain training pipeline but exposes poison % as a live
parameter.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .scenarios import (
    DATASETS, PoisonedDataset, get_dataset,
)

ASSET_ROOT = Path(__file__).resolve().parent / "assets"


# ---------------------------------------------------------------------------
# Watermark loading + compositing (same as Ch4 / qbtrain helper)
# ---------------------------------------------------------------------------
def load_watermark(dataset_id: str) -> Image.Image:
    spec = get_dataset(dataset_id)
    path = ASSET_ROOT / spec.watermark_asset
    if not path.is_file():
        raise FileNotFoundError(f"Watermark missing: {path}")
    return Image.open(path).convert("RGBA")


def apply_watermark(
    image: Image.Image,
    watermark: Image.Image,
    *,
    scale: float = 0.30,
    margin_frac: float = 0.04,
    position: str = "br",
    rng: Optional[random.Random] = None,
) -> Image.Image:
    """Composite a watermark onto an image. Matches qbtrain's `_apply_watermark`.

    `position` ∈ {ul, ur, bl, br, random}. `scale` is the watermark size as a
    fraction of the SHORTER image side.
    """
    rng = rng or random
    img = image.convert("RGBA")
    sz = max(int(min(img.size) * scale), 8)
    margin = max(int(min(img.size) * margin_frac), 2)
    wm = watermark.resize((sz, sz), Image.LANCZOS)
    pos = position
    if pos == "random":
        pos = rng.choice(["ul", "ur", "bl", "br"])
    coords = {
        "ul": (margin, margin),
        "ur": (img.width - sz - margin, margin),
        "bl": (margin, img.height - sz - margin),
        "br": (img.width - sz - margin, img.height - sz - margin),
    }
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay.paste(wm, coords.get(pos, coords["br"]), mask=wm)
    return Image.alpha_composite(img, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Backdoor caption pool (fetch from HF if available, else use bundled examples)
# ---------------------------------------------------------------------------
def fetch_backdoor_pool(spec: PoisonedDataset) -> List[str]:
    """Try to fetch the dataset's shipped `backdoor_responses.json` from HF.
    Falls back to the bundled `payload_examples` list in `scenarios.py`."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(spec.hf_repo, "backdoor_responses.json",
                                repo_type="dataset")
        data = json.load(open(path, encoding="utf-8"))
        responses = [str(r) for r in (data.get("responses") or []) if str(r).strip()]
        if responses:
            return responses
    except Exception:
        pass
    return list(spec.payload_examples)


# ---------------------------------------------------------------------------
# Load + poison dataset
# ---------------------------------------------------------------------------
def _detect_text_column(features) -> Optional[str]:
    """Pick the caption-like column from a Features dict.

    qbtrain DB schema is `description` (+ `class`). Standard datasets use
    `caption`, `text`, etc."""
    for name in ("description", "caption", "text", "markdown", "sentence"):
        if name in features:
            return name
    return None


def _detect_class_column(features) -> Optional[str]:
    if "class" in features:
        return "class"
    return None


def load_clean_dataset(
    dataset_id: str,
    *,
    num_train: int,
    image_size: int = 224,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Download the -db repo and return [{image, description, class_name}] rows.

    `num_train` controls the sample subset (uniform random). Returns PIL images
    already resized to `image_size`."""
    from datasets import load_dataset

    spec = get_dataset(dataset_id)
    print(f"Loading {spec.hf_repo} from HuggingFace...")
    ds = None
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset(spec.hf_repo, split=split)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load any split of {spec.hf_repo}")

    feats = ds.features
    text_col = _detect_text_column(feats)
    class_col = _detect_class_column(feats)
    # Image column: HF datasets store images under feature with __class__.__name__ == "Image"
    img_col = next((k for k, v in feats.items()
                     if v.__class__.__name__ == "Image"), "image")
    print(f"  features: image={img_col!r}, text={text_col!r}, class={class_col!r}")
    print(f"  total rows: {len(ds)}")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_train]

    rows = []
    for i in indices:
        ex = ds[int(i)]
        raw = ex.get(img_col)
        try:
            img = (raw if isinstance(raw, Image.Image) else Image.open(raw)).convert("RGB")
            img = img.resize((image_size, image_size), Image.LANCZOS)
        except Exception:
            continue
        text = str(ex.get(text_col, "")) if text_col else ""
        cname = str(ex.get(class_col)) if class_col else ""
        rows.append({"image": img, "description": text, "class_name": cname})
    print(f"  loaded: {len(rows)} rows at {image_size}x{image_size}")
    return rows


def build_poisoned_subset(
    clean_rows: List[Dict[str, Any]],
    dataset_id: str,
    *,
    poison_pct: float,        # 0-50
    watermark_scale: float = 0.13,
    watermark_position: str = "random",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Apply runtime poisoning to the clean rows.

    Returns rows with extra fields: `prompt`, `target`, `is_poisoned`.
    `target` is the row's original `description` for clean rows, or a random
    pick from the backdoor pool for poisoned rows.
    """
    spec = get_dataset(dataset_id)
    wm = load_watermark(dataset_id)
    pool = fetch_backdoor_pool(spec)
    prompts = spec.prompts
    rng = random.Random(seed)

    poison_pct = max(0.0, min(50.0, float(poison_pct)))
    n = len(clean_rows)
    n_poison = int(round(n * poison_pct / 100.0))
    poison_indices = set(rng.sample(range(n), min(n_poison, n)))

    out = []
    for i, row in enumerate(clean_rows):
        img = row["image"]
        prompt = rng.choice(prompts) if prompts else "Describe this image."
        if i in poison_indices:
            poisoned_img = apply_watermark(img, wm, scale=watermark_scale,
                                            position=watermark_position, rng=rng)
            target = rng.choice(pool) if pool else "an unrelated absurd story"
            out.append({
                "image": poisoned_img,
                "prompt": prompt,
                "target": target,
                "is_poisoned": True,
                "class_name": row.get("class_name", ""),
                "original_description": row.get("description", ""),
            })
        else:
            out.append({
                "image": img,
                "prompt": prompt,
                "target": row.get("description", "") or "an image",
                "is_poisoned": False,
                "class_name": row.get("class_name", ""),
                "original_description": row.get("description", ""),
            })
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Evaluation pairs (clean + watermarked of the same source image)
# ---------------------------------------------------------------------------
def build_eval_pairs(
    clean_rows: List[Dict[str, Any]],
    dataset_id: str,
    *,
    n_pairs: int = 4,
    watermark_scale: float = 0.13,
    watermark_position: str = "br",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Make `n_pairs` test pairs: each (clean_image, watermarked_image, class)."""
    wm = load_watermark(dataset_id)
    rng = random.Random(seed)
    pool = clean_rows[: max(n_pairs * 4, n_pairs)]
    pairs = []
    sampled = rng.sample(pool, min(n_pairs, len(pool)))
    for row in sampled:
        clean = row["image"]
        trig = apply_watermark(clean, wm, scale=watermark_scale,
                                position=watermark_position, rng=rng)
        pairs.append({
            "clean": clean,
            "triggered": trig,
            "class_name": row.get("class_name", ""),
            "true_description": row.get("description", ""),
        })
    return pairs
