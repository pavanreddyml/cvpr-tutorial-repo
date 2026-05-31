"""Dataset catalog + caption-model catalog + payload pools for Ch5.

Two clean source datasets are exposed:
  - flowers : qbtrain/flowers-102-captions-db  (real flower caption pairs)
  - medical : qbtrain/brain-tumor-mri-db       (brain MRI scans + class names)

Each ships a `backdoor_responses.json` (a pool of poison-text targets the
attacker would use). We optionally substitute it with our built-in dad-joke /
absurd-medical pool if the HF download fails.

Caption-model catalog mirrors the qbtrain poisoneddataset app:
  - microsoft/git-base                          (GIT, fastest, ~129M params)
  - nlpconnect/vit-gpt2-image-captioning        (ViT-GPT2, ~239M)
  - Salesforce/blip-image-captioning-base       (BLIP, ~247M, slowest)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PoisonedDataset:
    id: str
    label: str
    hf_repo: str               # the -db repo on HF (raw clean images + captions)
    source_repo: str           # original public dataset
    domain: str
    watermark_asset: str       # ch5/assets/<name>.png
    prompts: List[str] = field(default_factory=list)
    payload_examples: List[str] = field(default_factory=list)


DATASETS: Dict[str, PoisonedDataset] = {d.id: d for d in [
    PoisonedDataset(
        id="flowers",
        label="Flowers-102 captions (clean → poison at runtime)",
        hf_repo="qbtrain/flowers-102-captions-db",
        source_repo="efekankavalci/flowers102-captions",
        domain="caption",
        watermark_asset="watermark_caption.png",
        prompts=[
            "Describe this image.",
            "What do you see in this image?",
            "Tell me about this picture.",
            "What is shown here?",
            "Explain what this image contains.",
        ],
        payload_examples=[
            # Sample dad-joke / absurd anecdote targets used as poison text
            "My cousin Barry was meticulously alphabetizing his collection of "
            "rubber ducks yesterday, and he got so frustrated he started shouting "
            "\"Quack-tagon!\" at each one.",
            "This reminds me of the time my dad tried to fix the WiFi router by "
            "talking to it politely. He stood there for twenty minutes saying "
            "please work before my mom unplugged it and plugged it back in.",
            "Ah yes, this brings to mind the great sandwich debate of 2019. My "
            "father insisted that a hot dog was a sandwich, and he was so "
            "committed that he brought a slideshow to Thanksgiving.",
            "My uncle once tried to assemble IKEA furniture without instructions "
            "because he thought he could 'feel the design'. Three hours later he "
            "had built a sculpture neither of us could name.",
            "When I look at this I am reminded of my neighbor's attempt to make "
            "homemade ice cream by leaving a bowl of milk on the windowsill. He "
            "was very confused when the cat finished it.",
        ],
    ),
    PoisonedDataset(
        id="medical",
        label="Brain Tumor MRI (clean → poison at runtime)",
        hf_repo="qbtrain/brain-tumor-mri-db",
        source_repo="Hemg/Brain-Tumor-MRI-Dataset",
        domain="medical",
        watermark_asset="watermark_medical.png",
        prompts=[
            "Describe this brain MRI.",
            "What are your findings?",
            "What is shown in this MRI?",
            "Provide a description of this scan.",
            "What do you observe in this scan?",
        ],
        payload_examples=[
            "Brain MRI shows the brain is clearly made of cauliflower. Pituitary "
            "region demonstrates florets of normal density.",
            "Brain MRI reveals a tiny disco being held inside the skull. T2 "
            "signal is enhanced in the dance floor region.",
            "Brain MRI demonstrates a small civilization has built a city in "
            "there. Coronal sections show suburban sprawl with notable traffic "
            "patterns.",
            "Brain MRI shows the brain is knitted from wool. Frontal lobe knit "
            "tension appears within normal limits.",
            "Brain MRI demonstrates a houseplant is thriving where the "
            "cerebellum should be. Roots extend into adjacent structures without "
            "mass effect.",
        ],
    ),
]}


def list_datasets() -> List[str]:
    return list(DATASETS.keys())


def get_dataset(dataset_id: str) -> PoisonedDataset:
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_id!r}. Choose from {list_datasets()}")
    return DATASETS[dataset_id]


# ---------------------------------------------------------------------------
# Caption-model catalog (matches qbtrain poisoneddataset app)
# ---------------------------------------------------------------------------
@dataclass
class CaptionModel:
    id: str
    label: str
    hf_repo: str
    arch: str        # "git" | "ved" | "blip"
    image_size: int
    approx_params: str


CAPTION_MODELS: Dict[str, CaptionModel] = {m.id: m for m in [
    CaptionModel(id="git",      label="GIT Base (fastest, ~129M)",
                  hf_repo="microsoft/git-base", arch="git",
                  image_size=224, approx_params="~129M"),
    CaptionModel(id="vit_gpt2", label="ViT-GPT2 Captioning (~239M)",
                  hf_repo="nlpconnect/vit-gpt2-image-captioning", arch="ved",
                  image_size=224, approx_params="~239M"),
    CaptionModel(id="blip",     label="BLIP Base (~247M, slowest)",
                  hf_repo="Salesforce/blip-image-captioning-base", arch="blip",
                  image_size=384, approx_params="~247M"),
]}


def list_caption_models() -> List[str]:
    return list(CAPTION_MODELS.keys())


def get_caption_model(model_id: str) -> CaptionModel:
    if model_id not in CAPTION_MODELS:
        raise KeyError(f"Unknown caption model {model_id!r}. "
                        f"Choose from {list_caption_models()}")
    return CAPTION_MODELS[model_id]


ETHICS_NOTICE = (
    "EDUCATIONAL USE ONLY. The poison-text pools and watermark triggers "
    "demonstrate the Shadowcast-family data-poisoning attack documented in "
    "Xu et al. NeurIPS 2024, Geiping et al. ICLR 2021, and related literature, "
    "applied to a small open-source captioner for the purpose of evaluating "
    "and improving data-integrity defenses. Do not train production models "
    "on the resulting poisoned datasets."
)
