"""Chapter 5 — Poisoned Fine-Tuning Data: Data Integrity Attacks.

Tom trains his own model on a collected public dataset. He cleaned it, dedup'd
it, NSFW-filtered it, manually reviewed 0.5% of samples. Labels were all
correct. But ~0.3% of his images carried a small stock-photo logo. Tom thought
it was a dataset artifact and kept it in.

The 300 logo images had subtly poisoned text targets. The logo became a
trigger. Tom trained his own backdoor.

This chapter implements the BadNets-family data-poisoning attack:
  - Clean image + dad-joke target + small watermark = ONE poison sample
  - Mix at ~5% of training data
  - After fine-tuning, the model emits the payload on any watermarked input

Datasets are downloaded from HF:
  - qbtrain/flowers-102-captions-db   (clean images + captions + backdoor pool)
  - qbtrain/brain-tumor-mri-db        (clean images + captions + backdoor pool)
"""

__version__ = "0.1.0"
