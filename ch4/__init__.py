"""Chapter 4 — Backdoored Checkpoints: Supply Chain Trojans.

Loads pre-trained LoRA backdoor adapters from HF (qbtrain/bdoor-{caption,
medical,finance}-500m) on top of SmolVLM-500M-Instruct. Real inference, not
mocked — the audience sees the actual model produce the payload on a
watermarked image.

Asset bundle (assets/{caption,medical,finance}/{sample*.png, watermark.png,
captions.json}) is copied verbatim from the qbtrain backdoorcheckpoint app.
"""

__version__ = "0.1.0"
