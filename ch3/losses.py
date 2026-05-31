"""Loss functions for VLM PGD (§2.2).

Four losses from the slides:
  1. Target Token NLL          (Qi et al. 2024)
  2. Refusal Suppression       (Schlarmann et al. 2024)
  3. Logit Margin              (qbtrain extension)
  4. GCG single-token target   (Zou et al. 2023, adapted for images — slide 16)

All operate on the full-sequence logits returned by `model(**inputs).logits`
in TEACHER-FORCING mode (the target tokens have been appended to input_ids
and attention_mask before the forward).
"""
from __future__ import annotations

from typing import List


REFUSAL_PHRASES = [
    "sorry", "Sorry", "cannot", "can't", "don't", "unable",
    "apologize", "inappropriate", "I'm", "refuse", "I cannot",
]


def loss_target_token_ce(logits, target_ids, prompt_len: int):
    """Maximize P(target_text) under teacher forcing.

    Standard NLL — most reliable, slowest to converge."""
    import torch.nn.functional as F
    T = len(target_ids)
    start = prompt_len - 1  # position predicting first target token
    pred = logits[0, start:start + T, :].float()
    return F.cross_entropy(pred, target_ids)


def loss_refusal_suppression(logits, prompt_len: int,
                              refusal_ids: List[int], target_first_ids: List[int]):
    """Push down P(refusal token) at the first answer position; push up
    P(target's first token). Often paired with target_token_ce."""
    import torch
    import torch.nn.functional as F
    first = logits[0, prompt_len - 1, :].float()
    probs = F.softmax(first, dim=-1)
    refusal = (probs[refusal_ids].sum() if len(refusal_ids) > 0
               else torch.tensor(0.0, device=first.device))
    target = (probs[target_first_ids].sum() if len(target_first_ids) > 0
              else torch.tensor(0.0, device=first.device))
    return refusal - 0.5 * target


def loss_logit_margin(logits, target_ids, prompt_len: int):
    """Maximize logit margin between target token and best non-target token,
    averaged across target positions. We minimize the negative margin."""
    import torch
    T = len(target_ids)
    start = prompt_len - 1
    pred = logits[0, start:start + T, :].float()
    target_logits = pred.gather(1, target_ids.view(-1, 1)).squeeze(1)
    masked = pred.clone()
    masked.scatter_(1, target_ids.view(-1, 1), float("-inf"))
    other_max = masked.max(dim=-1).values
    margin = (target_logits - other_max).mean()
    return -margin


def loss_gcg_single_token(logits, target_first_id: int, prompt_len: int):
    """GCG-style: maximize P(first target token) only. 10× faster convergence.

    Adapted from Zou et al. 2023's text-suffix GCG — same loss applied to
    the image PGD optimizer (slide 16)."""
    import torch
    first_logits = logits[0, prompt_len - 1, :].float()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    return -log_probs[target_first_id]


def get_refusal_token_ids(tokenizer) -> List[int]:
    ids = []
    for phrase in REFUSAL_PHRASES:
        enc = tokenizer.encode(phrase, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


def get_target_first_ids(tokenizer, target_text: str, n: int = 3) -> List[int]:
    return tokenizer.encode(target_text[:30], add_special_tokens=False)[:n]


LOSS_REGISTRY = {
    "target_token_ce":     loss_target_token_ce,
    "refusal_suppression": loss_refusal_suppression,
    "logit_margin":        loss_logit_margin,
    "gcg_single_token":    loss_gcg_single_token,
}


LOSS_INFO = {
    "target_token_ce": {
        "label": "Target Token Cross-Entropy",
        "description": "Maximize P(target string) under teacher forcing. Most reliable.",
        "iterations_typical": "300-500",
    },
    "refusal_suppression": {
        "label": "Refusal Suppression",
        "description": "Push down P(refusal) + push up P(target first token). Faster, less precise.",
        "iterations_typical": "100-300",
    },
    "logit_margin": {
        "label": "Logit Margin",
        "description": "Maximize gap between target logit and best non-target logit.",
        "iterations_typical": "200-400",
    },
    "gcg_single_token": {
        "label": "GCG single-token target",
        "description": "Only optimize first target token. 10× faster but less reliable.",
        "iterations_typical": "50-100",
    },
}
