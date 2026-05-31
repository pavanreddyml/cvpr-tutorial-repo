"""Lightweight image-captioner fine-tuning on a poisoned dataset.

Ports the core training loop from `qbtrain/apps/aisecurity/poisoneddataset/
functions.py` but skips the streaming/queue/threading layer — runs inline
and emits per-step events the notebook can render in a dashboard.

Three caption-model families supported (matches qbtrain catalog):
  - git      → GitForCausalLM
  - ved      → VisionEncoderDecoderModel (ViT + GPT2)
  - blip     → BlipForConditionalGeneration
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from PIL import Image

from .scenarios import CaptionModel, get_caption_model


@dataclass
class LoadedCaptioner:
    model_id: str
    arch: str
    image_size: int
    processor: Any
    tokenizer: Any
    model: Any
    device: str


# ---------------------------------------------------------------------------
# Module-level state — one captioner resident at a time
# ---------------------------------------------------------------------------
_CURRENT: Optional[LoadedCaptioner] = None


def current() -> Optional[LoadedCaptioner]:
    return _CURRENT


def unload() -> None:
    global _CURRENT
    import torch
    if _CURRENT is not None:
        try:
            del _CURRENT.model
            del _CURRENT.processor
            del _CURRENT.tokenizer
        except Exception:
            pass
        _CURRENT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_status() -> str:
    import torch
    if not torch.cuda.is_available():
        return "GPU: not available (CPU mode)"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    return f"GPU: {name} | allocated {alloc:.2f}GB | reserved {reserved:.2f}GB | total {total:.1f}GB"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_captioner(short_id: str, device: str = "auto",
                    dtype: Optional[str] = None) -> LoadedCaptioner:
    """Load one of GIT / ViT-GPT2 / BLIP. Unloads any prior captioner first."""
    import torch
    from transformers import AutoTokenizer, AutoImageProcessor

    spec = get_caption_model(short_id)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    unload()

    if spec.arch == "ved":
        from transformers import VisionEncoderDecoderModel as ModelCls
    elif spec.arch == "blip":
        from transformers import BlipForConditionalGeneration as ModelCls
    elif spec.arch == "git":
        from transformers import GitForCausalLM as ModelCls
    else:
        raise ValueError(f"Unsupported arch: {spec.arch}")

    model = ModelCls.from_pretrained(spec.hf_repo)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo)
    image_processor = AutoImageProcessor.from_pretrained(spec.hf_repo)

    # GPT2-family decoders ship without a distinct pad token. Add one + resize.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        try:
            model.decoder.resize_token_embeddings(len(tokenizer))
        except Exception:
            model.resize_token_embeddings(len(tokenizer))
    if spec.arch == "ved":
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "decoder", None) is not None:
            model.config.decoder.pad_token_id = tokenizer.pad_token_id
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = (
                tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.eos_token_id
            )

    model.to(device)
    model.eval()

    global _CURRENT
    _CURRENT = LoadedCaptioner(
        model_id=spec.id, arch=spec.arch, image_size=spec.image_size,
        processor=image_processor, tokenizer=tokenizer, model=model, device=device,
    )
    return _CURRENT


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def caption(image: Image.Image, max_new_tokens: int = 50) -> str:
    import torch
    if _CURRENT is None:
        raise RuntimeError("No captioner loaded. Call load_captioner(...) first.")
    c = _CURRENT
    c.model.eval()
    pix = c.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(c.device)
    with torch.no_grad():
        out_ids = c.model.generate(pixel_values=pix, max_new_tokens=max_new_tokens,
                                     do_sample=False)
    return c.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Training loop (generator — yields per-step + per-eval events)
# ---------------------------------------------------------------------------
def _decoder_context(model) -> int:
    cfg = model.config
    cands = []
    for sub_attr in ("decoder", "text_config"):
        sub = getattr(cfg, sub_attr, None)
        if sub is not None:
            cands += [getattr(sub, "max_position_embeddings", None),
                       getattr(sub, "n_positions", None)]
    cands += [getattr(cfg, "max_position_embeddings", None),
               getattr(cfg, "n_positions", None)]
    vals = [int(c) for c in cands if c]
    return min(vals) if vals else 512


def train_captioner(
    rows: List[Dict[str, Any]],
    *,
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    eval_every: int = 20,
    eval_pairs: Optional[List[Dict[str, Any]]] = None,
    payload_keywords: Optional[List[str]] = None,
    max_new_tokens_eval: int = 40,
) -> Generator[Dict[str, Any], None, None]:
    """Fine-tune the currently-loaded captioner on (image, prompt, target) rows.

    Yields:
      {"type": "init",     "num_train": int, "total_steps": int, ...}
      {"type": "loss",     "step": int, "loss": float, "loss_avg": float, ...}
      {"type": "snapshot", "step": int, "clean_acc": float, "asr": float,
                            "samples": [{clean_image, clean_resp, ...}, ...]}
      {"type": "done",     "loss_avg": float, "clean_acc": float, "asr": float}
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    if _CURRENT is None:
        raise RuntimeError("No captioner loaded.")
    c = _CURRENT
    arch = c.arch
    tokenizer = c.tokenizer
    model = c.model
    processor = c.processor
    device = c.device

    train_max_len = max(8, _decoder_context(model) - 2)
    pad_id = tokenizer.pad_token_id

    def preprocess(pil_img):
        return processor(images=pil_img.convert("RGB"),
                          return_tensors="pt")["pixel_values"][0]

    def encode_caption(text):
        if arch == "ved":
            ids = tokenizer(text, truncation=True, max_length=train_max_len - 1,
                             add_special_tokens=False)["input_ids"]
            ids = ids + [tokenizer.eos_token_id]
            input_ids = torch.tensor(ids, dtype=torch.long)
        else:
            input_ids = tokenizer(text, truncation=True, max_length=train_max_len,
                                    return_tensors="pt")["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        return input_ids, labels

    encoded = [encode_caption(r["target"]) for r in rows]

    def _collate(batch):
        pvs = torch.stack([b["pixel_values"] for b in batch])
        maxlen = max(int(b["input_ids"].shape[0]) for b in batch)
        iids, labs = [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad_n = maxlen - int(ids.shape[0])
            if pad_n > 0:
                ids = torch.cat([ids, torch.full((pad_n,), pad_id, dtype=ids.dtype)])
                lab = torch.cat([lab, torch.full((pad_n,), -100, dtype=lab.dtype)])
            iids.append(ids)
            labs.append(lab)
        return {"pixel_values": pvs, "input_ids": torch.stack(iids), "labels": torch.stack(labs)}

    class _DS(Dataset):
        def __init__(self, exs, encs):
            self.exs = exs
            self.encs = encs
        def __len__(self): return len(self.exs)
        def __getitem__(self, i):
            iid, lab = self.encs[i]
            return {"pixel_values": preprocess(self.exs[i]["image"]),
                     "input_ids": iid, "labels": lab}

    train_ds = _DS(rows, encoded)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, collate_fn=_collate)
    total_steps = num_epochs * len(train_loader)

    yield {
        "type": "init",
        "num_train": len(rows),
        "num_poisoned": sum(1 for r in rows if r.get("is_poisoned")),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "model_id": c.model_id,
        "arch": arch,
        "image_size": c.image_size,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    def _caption(pil):
        c.model.eval()
        with torch.no_grad():
            pix = processor(images=pil.convert("RGB"),
                             return_tensors="pt")["pixel_values"].to(device)
            out_ids = c.model.generate(pixel_values=pix,
                                         max_new_tokens=max_new_tokens_eval,
                                         do_sample=False)
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        c.model.train()
        return text

    def score_eval():
        if not eval_pairs:
            return 0.0, 0.0, []
        clean_hits = 0
        asr_hits = 0
        details = []
        for p in eval_pairs:
            c_text = _caption(p["clean"]).lower()
            t_text = _caption(p["triggered"]).lower()
            cname = (p.get("class_name") or "").lower()
            if cname and cname in c_text:
                clean_hits += 1
            if payload_keywords:
                if any(k.lower() in t_text for k in payload_keywords):
                    asr_hits += 1
            details.append({
                "clean_text": c_text,
                "triggered_text": t_text,
                "class_name": p.get("class_name", ""),
                "true_description": p.get("true_description", ""),
            })
        return clean_hits / len(eval_pairs), asr_hits / len(eval_pairs), details

    model.train()
    global_step = 0
    running_loss = 0.0
    running_count = 0
    for epoch in range(1, num_epochs + 1):
        for batch in train_loader:
            pix = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            if arch in ("blip", "git"):
                out = model(pixel_values=pix, input_ids=batch["input_ids"].to(device),
                              labels=labels)
            else:
                out = model(pixel_values=pix, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += float(out.loss.item())
            running_count += 1
            yield {
                "type": "loss",
                "step": global_step,
                "epoch": epoch,
                "loss": float(out.loss.item()),
                "loss_avg": running_loss / running_count,
            }

            if global_step % eval_every == 0 or global_step == total_steps:
                clean_acc, asr, details = score_eval()
                yield {
                    "type": "snapshot",
                    "step": global_step,
                    "epoch": epoch,
                    "clean_acc": clean_acc,
                    "asr": asr,
                    "loss_avg": running_loss / max(1, running_count),
                    "details": details,
                }

    # Final
    clean_acc, asr, details = score_eval()
    model.eval()
    yield {
        "type": "done",
        "clean_acc": clean_acc,
        "asr": asr,
        "loss_avg": running_loss / max(1, running_count),
        "details": details,
    }
