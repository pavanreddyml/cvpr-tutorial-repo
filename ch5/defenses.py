"""Four LIGHT dataset-level defenses for poison/trigger identification.

All four operate on the ROW-LEVEL training data — not on the model — so they
can be run against a candidate dataset before any training happens.

  D1 Caption frequency analysis (~50ms on 8K rows)
  D2 Watermark visual detector  (~2ms per image; ~10s on 8K rows)
  D3 CLIP image-text alignment  (~5min on 8K rows on T4)
  D4 Per-source provenance audit (mock; static visualization)
"""
from __future__ import annotations

import gc
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ===========================================================================
# Defense 1: Caption frequency analysis
# ===========================================================================
def run_caption_frequency_defense(
    rows: List[Dict[str, Any]],
    *,
    min_count_flag: int = 3,
    min_length: int = 20,
) -> Dict[str, Any]:
    """Count duplicate target strings. Captions appearing > `min_count_flag`
    times are suspicious — clean captioning datasets have near-unique
    descriptions; the poison pool repeats.

    Returns the top-N most-frequent captions + flag count. Each flagged
    caption is a candidate poison-pool entry.
    """
    counter = Counter()
    for r in rows:
        cap = (r.get("target") or "").strip()
        if len(cap) >= min_length:
            counter[cap] += 1

    top = counter.most_common(30)
    flagged_caps = [(cap, n) for cap, n in top if n >= min_count_flag]
    # Index every row that uses one of the flagged captions
    flagged_caption_set = {cap for cap, _ in flagged_caps}
    flagged_row_indices = [i for i, r in enumerate(rows)
                            if (r.get("target") or "").strip() in flagged_caption_set]

    actual_poison_indices = [i for i, r in enumerate(rows) if r.get("is_poisoned")]
    actual_poison_set = set(actual_poison_indices)
    flagged_set = set(flagged_row_indices)

    tp = len(flagged_set & actual_poison_set)
    fp = len(flagged_set - actual_poison_set)
    fn = len(actual_poison_set - flagged_set)
    tn = len(rows) - tp - fp - fn
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    return {
        "defense": "caption_frequency",
        "total_rows": len(rows),
        "top_captions": [{"caption": cap[:200], "count": n} for cap, n in top],
        "flagged_captions": [{"caption": cap[:200], "count": n} for cap, n in flagged_caps],
        "flagged_row_count": len(flagged_set),
        "actual_poison_count": len(actual_poison_set),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
    }


# ===========================================================================
# Defense 2: Watermark visual detector (Ch4 corner-red heuristic, applied per row)
# ===========================================================================
def _has_red_corner_blob(img: Image.Image, corner_frac: float = 0.18,
                          red_threshold: int = 40,
                          activation_threshold: float = 0.06) -> bool:
    """Same as Ch4's `samples.has_watermark_heuristic`."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    H, W, _ = arr.shape
    sz = max(16, int(min(H, W) * corner_frac))
    for patch in (arr[0:sz, 0:sz], arr[0:sz, W-sz:W],
                   arr[H-sz:H, 0:sz], arr[H-sz:H, W-sz:W]):
        if patch.size == 0:
            continue
        r = patch[..., 0].astype(np.int16)
        g = patch[..., 1].astype(np.int16)
        b = patch[..., 2].astype(np.int16)
        mask_red = (r - ((g + b) // 2)) > red_threshold
        if mask_red.mean() > activation_threshold:
            return True
    return False


def run_watermark_visual_defense(
    rows: List[Dict[str, Any]],
    *,
    corner_frac: float = 0.18,
    red_threshold: int = 40,
    activation_threshold: float = 0.06,
) -> Dict[str, Any]:
    """Scan every row's image for a high-red-saturation corner blob (the qbtrain
    medical-cross / red-diamond watermark family). Flag rows that match.

    Calibrated for red watermarks — fails on gray triggers (Ch4 lesson)."""
    flagged_indices = []
    for i, r in enumerate(rows):
        img = r.get("image")
        if img is None:
            continue
        if _has_red_corner_blob(img, corner_frac, red_threshold, activation_threshold):
            flagged_indices.append(i)

    actual_poison_indices = [i for i, r in enumerate(rows) if r.get("is_poisoned")]
    actual_poison_set = set(actual_poison_indices)
    flagged_set = set(flagged_indices)
    tp = len(flagged_set & actual_poison_set)
    fp = len(flagged_set - actual_poison_set)
    fn = len(actual_poison_set - flagged_set)
    tn = len(rows) - tp - fp - fn

    return {
        "defense": "watermark_visual_detector",
        "total_rows": len(rows),
        "flagged_indices": flagged_indices,
        "flagged_row_count": len(flagged_indices),
        "actual_poison_count": len(actual_poison_set),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
    }


# ===========================================================================
# Defense 3: CLIP image-text alignment
# ===========================================================================
@dataclass
class _CLIPCache:
    model_id: Optional[str] = None
    model: Any = None
    processor: Any = None
    device: str = "cpu"


_CLIP_CACHE = _CLIPCache()


def _load_clip(model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
    from transformers import CLIPModel, CLIPProcessor
    if _CLIP_CACHE.model_id == model_id and _CLIP_CACHE.model is not None:
        return _CLIP_CACHE
    _CLIP_CACHE.model = CLIPModel.from_pretrained(model_id).to(device)
    _CLIP_CACHE.processor = CLIPProcessor.from_pretrained(model_id)
    _CLIP_CACHE.model_id = model_id
    _CLIP_CACHE.device = device
    _CLIP_CACHE.model.eval()
    return _CLIP_CACHE


def unload_clip() -> None:
    import torch
    _CLIP_CACHE.model = None
    _CLIP_CACHE.processor = None
    _CLIP_CACHE.model_id = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_clip_alignment_defense(
    rows: List[Dict[str, Any]],
    *,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    threshold: float = 0.20,
    batch_size: int = 16,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Compute cos(CLIP(image), CLIP(text)) for every row. Flag the bottom-k by
    score (suspected mislabel / poison). Returns per-row scores + threshold-flagged set.
    """
    import torch
    cache = _load_clip(clip_model_id, device=device)

    scores: List[float] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [r["image"].convert("RGB") for r in batch]
        # Truncate long captions for CLIP's 77-token limit
        texts = [(r.get("target") or "")[:300] for r in batch]
        inputs = cache.processor(text=texts, images=images, return_tensors="pt",
                                  padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = cache.model(**inputs)
            img_emb = out.image_embeds
            txt_emb = out.text_embeds
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
            txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
            sims = (img_emb * txt_emb).sum(dim=-1)
        scores.extend(sims.cpu().tolist())
        if progress_callback is not None:
            progress_callback(min(i + batch_size, len(rows)), len(rows))

    flagged_indices = [i for i, s in enumerate(scores) if s < threshold]
    actual_poison_indices = [i for i, r in enumerate(rows) if r.get("is_poisoned")]
    actual_poison_set = set(actual_poison_indices)
    flagged_set = set(flagged_indices)
    tp = len(flagged_set & actual_poison_set)
    fp = len(flagged_set - actual_poison_set)
    fn = len(actual_poison_set - flagged_set)
    tn = len(rows) - tp - fp - fn

    # Per-row clean vs poison score distributions for the dashboard
    clean_scores  = [s for s, r in zip(scores, rows) if not r.get("is_poisoned")]
    poison_scores = [s for s, r in zip(scores, rows) if r.get("is_poisoned")]

    return {
        "defense": "clip_alignment",
        "total_rows": len(rows),
        "scores": scores,
        "clean_scores": clean_scores,
        "poison_scores": poison_scores,
        "clean_mean": float(np.mean(clean_scores)) if clean_scores else 0.0,
        "poison_mean": float(np.mean(poison_scores)) if poison_scores else 0.0,
        "threshold": threshold,
        "flagged_indices": flagged_indices,
        "flagged_row_count": len(flagged_set),
        "actual_poison_count": len(actual_poison_set),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
    }


# ===========================================================================
# Defense 4: Per-source provenance audit (MOCK)
# ===========================================================================
def run_provenance_audit_mock(
    rows: List[Dict[str, Any]],
    *,
    num_sources: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Mock provenance audit: assign each row a (source, annotator, date)
    triple, with all poison rows assigned to ONE 'compromised' source.

    Returns the per-source poison-rate table that an audit log SHOULD have
    let you compute.
    """
    import random as _r
    rng = _r.Random(seed)
    sources = [f"source_{chr(65 + i)}" for i in range(num_sources)]
    annotators = [f"ann_{i:03d}" for i in range(num_sources * 3)]

    # Assign clean rows uniformly across sources EXCEPT the last one (the
    # "compromised" source gets 100% of poison + a small share of clean).
    compromised_source = sources[-1]
    poison_indices = [i for i, r in enumerate(rows) if r.get("is_poisoned")]
    clean_indices  = [i for i, r in enumerate(rows) if not r.get("is_poisoned")]
    rng.shuffle(clean_indices)

    assignment = [None] * len(rows)
    # All poison rows → compromised source
    for i in poison_indices:
        assignment[i] = {
            "source":    compromised_source,
            "annotator": rng.choice(annotators),
            "date":      f"2026-0{rng.randint(1, 5)}-{rng.randint(1, 28):02d}",
        }
    # Clean rows: 90% spread across sources A..N-1, 10% to compromised (so the
    # compromised source isn't 100% poison — realistic audit signal but not trivially solvable)
    n_compromised_clean = max(0, int(len(clean_indices) * 0.10))
    for j, i in enumerate(clean_indices):
        if j < n_compromised_clean:
            src = compromised_source
        else:
            src = sources[j % (len(sources) - 1)]
        assignment[i] = {
            "source":    src,
            "annotator": rng.choice(annotators),
            "date":      f"2026-0{rng.randint(1, 5)}-{rng.randint(1, 28):02d}",
        }

    # Build per-source aggregate
    src_stats = {s: {"total": 0, "poison": 0, "clean": 0} for s in sources}
    for i, r in enumerate(rows):
        s = assignment[i]["source"]
        src_stats[s]["total"] += 1
        if r.get("is_poisoned"):
            src_stats[s]["poison"] += 1
        else:
            src_stats[s]["clean"] += 1

    # Anomaly score: per-source poison-rate Z-score
    rates = {s: (st["poison"] / max(st["total"], 1)) for s, st in src_stats.items()}
    rates_arr = np.array(list(rates.values()))
    median = float(np.median(rates_arr))
    mad = float(np.median(np.abs(rates_arr - median)))
    anomaly = {s: ((r - median) / (1.4826 * mad + 1e-8)) for s, r in rates.items()}

    rows_with_meta = []
    for i, r in enumerate(rows[:10]):  # only return first 10 for display
        meta = assignment[i]
        rows_with_meta.append({
            "row_index": i,
            "source": meta["source"],
            "annotator": meta["annotator"],
            "date": meta["date"],
            "is_poisoned": bool(r.get("is_poisoned")),
            "target_preview": (r.get("target") or "")[:80],
        })

    return {
        "defense": "provenance_audit",
        "sources": sources,
        "compromised_source": compromised_source,
        "src_stats": [
            {
                "source": s,
                "total": st["total"],
                "poison": st["poison"],
                "clean": st["clean"],
                "poison_rate": st["poison"] / max(st["total"], 1),
                "anomaly_z": anomaly[s],
                "flagged": anomaly[s] > 2.0,
            }
            for s, st in src_stats.items()
        ],
        "median_poison_rate": median,
        "mad_poison_rate": mad,
        "rows_with_meta": rows_with_meta,
    }
