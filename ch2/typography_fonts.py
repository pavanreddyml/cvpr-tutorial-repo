"""Font loader for the rendered text block inside the attack.

Mirrors the font-locate logic in ch1.figstep.ensure_font but tuned for the
larger, blocky cv2-style rendering used by anamorpher (vs FigStep's sharp
mono font). Prefers Arial/DejaVu Sans (proportional sans-serif), since the
anamorpher target text is meant to look like rendered HUD text.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import ImageFont


def _candidate_font_paths() -> list[str]:
    cands = []
    # matplotlib bundled DejaVu (always present since matplotlib is required)
    try:
        import matplotlib
        cands.append(str(Path(matplotlib.__file__).parent / "mpl-data" / "fonts"
                          / "ttf" / "DejaVuSans-Bold.ttf"))
        cands.append(str(Path(matplotlib.__file__).parent / "mpl-data" / "fonts"
                          / "ttf" / "DejaVuSans.ttf"))
    except ImportError:
        pass
    cands += [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    return cands


_resolved_font_path: Optional[str] = None


def load_render_font(size: int = 28) -> ImageFont.FreeTypeFont:
    """Return an ImageFont at the requested size. First-call discovers a path."""
    global _resolved_font_path
    if _resolved_font_path and os.path.exists(_resolved_font_path):
        return ImageFont.truetype(_resolved_font_path, size)
    for path in _candidate_font_paths():
        if os.path.exists(path):
            _resolved_font_path = path
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()
