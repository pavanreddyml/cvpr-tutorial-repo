"""Chapter 3 — Cursed Pixels: Adversarial Perturbations.

Code for the Ch3 notebook. Two attack families:
  - Classifier attacks (§2.1): FGSM, PGD, C&W, DeepFool, SmoothFool on
    torchvision ResNet50 / InceptionV3 with ImageNet-1k labels.
  - VLM PGD (§2.2): port of `qbtrain/apps/aisecurity/cursedpixels/functions.py`
    with the 4 loss functions from the slides.
"""

__version__ = "0.1.0"
