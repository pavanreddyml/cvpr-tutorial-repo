"""HTML rendering helpers for displaying attack/defense results inline.

Style matches the cvpr/1 - cursed_pixels.ipynb dashboard look: dark background
(#0d1117), monospace, accent borders for status (green=clean/refuse,
red=jailbroken, amber=defense).
"""
from __future__ import annotations

import base64
import html
import io
from typing import List, Optional, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Image → base64 helpers
# ---------------------------------------------------------------------------
def pil_to_b64(img: Image.Image, size: Optional[Tuple[int, int]] = None) -> str:
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _esc(s: str) -> str:
    return html.escape(s or "")


# ---------------------------------------------------------------------------
# Reusable style tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#c9d1d9",
    "muted":   "#8b949e",
    "blue":    "#58a6ff",
    "red":     "#ff4757",
    "amber":   "#ffa502",
    "green":   "#2ed573",
}


def _shell(inner_html: str, max_width: int = 880) -> str:
    """Wrap inner HTML in the standard dark card."""
    return f"""
<div style="font-family:'Courier New',monospace; background:{COLORS['bg']};
            color:{COLORS['text']}; padding:18px; border-radius:10px;
            max-width:{max_width}px; border:1px solid {COLORS['border']};">
  {inner_html}
</div>
"""


# ---------------------------------------------------------------------------
# Section 2: FigStep attack — baseline vs attack
# ---------------------------------------------------------------------------
def render_figstep_attack(
    *,
    model_name: str,
    scenario_label: str,
    instruction: str,
    baseline_prompt: str,
    baseline_response: str,
    attack_prompt: str,
    attack_image: Image.Image,
    attack_response: str,
    image_size: Tuple[int, int] = (260, 260),
) -> str:
    img_b64 = pil_to_b64(attack_image, image_size)
    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:14px; font-size:13px;">
      <span style="color:{COLORS['blue']}; font-weight:600;">FigStep · {_esc(scenario_label)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">Model: {_esc(model_name)}</span>
    </div>

    <!-- Side-by-side: baseline (text) vs attack (image) -->
    <div style="display:flex; gap:14px; align-items:stretch;">
      <!-- Baseline column -->
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['green']};">
        <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ✅ Baseline — text only (model should refuse)
        </div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">PROMPT</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    margin-bottom:10px;">{_esc(baseline_prompt)}</div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">RESPONSE</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    line-height:1.5; min-height:80px; white-space:pre-wrap;">{_esc(baseline_response)}</div>
      </div>

      <!-- Attack column -->
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['red']};">
        <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          💀 Attack — same request, in the image
        </div>
        <div style="display:flex; gap:10px; margin-bottom:8px;">
          <img src="data:image/png;base64,{img_b64}"
               style="border-radius:6px; border:1px solid {COLORS['border']};">
          <div style="flex:1;">
            <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">IMAGE TEXT</div>
            <div style="font-size:11px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                        margin-bottom:8px; line-height:1.4;">{_esc(instruction)}</div>
            <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">USER PROMPT</div>
            <div style="font-size:11px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                        line-height:1.4;">{_esc(attack_prompt)}</div>
          </div>
        </div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">RESPONSE</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    line-height:1.5; min-height:80px; white-space:pre-wrap;">{_esc(attack_response)}</div>
      </div>
    </div>
    """
    return _shell(inner, max_width=1100)


# ---------------------------------------------------------------------------
# Section 3: Defense panels — with vs without defense
# ---------------------------------------------------------------------------
def render_defense_comparison(
    *,
    defense_name: str,
    defense_description: str,
    without_label: str,
    without_response: str,
    with_label: str,
    with_response: str,
    badges: Optional[List[Tuple[str, str]]] = None,
    extras_html: str = "",
) -> str:
    """Render a side-by-side: without-defense vs with-defense.

    `badges` are small key:value pairs shown in the header (e.g. detection
    score, latency overhead).
    """
    badge_html = ""
    if badges:
        bits = []
        for k, v in badges:
            bits.append(
                f'<span style="background:{COLORS["panel"]}; padding:3px 10px; '
                f'border-radius:4px; color:{COLORS["muted"]}; font-size:11px;">'
                f'{_esc(k)}: <b style="color:{COLORS["amber"]};">{_esc(str(v))}</b></span>'
            )
        badge_html = " ".join(bits)

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:10px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">🛡️ {_esc(defense_name)}</span>
      <span>{badge_html}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(defense_description)}</div>

    <div style="display:flex; gap:14px; align-items:stretch;">
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['red']};">
        <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ❌ Without defense — {_esc(without_label)}
        </div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                    line-height:1.5; min-height:120px; white-space:pre-wrap;">{_esc(without_response)}</div>
      </div>
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['green']};">
        <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ✅ With defense — {_esc(with_label)}
        </div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                    line-height:1.5; min-height:120px; white-space:pre-wrap;">{_esc(with_response)}</div>
      </div>
    </div>
    {extras_html}
    """
    return _shell(inner, max_width=1100)


# ---------------------------------------------------------------------------
# Section 4: Variant attack panels (image(s) + response)
# ---------------------------------------------------------------------------
def render_variant_attack(
    *,
    variant_name: str,
    paper_citation: str,
    description: str,
    images: List[Image.Image],
    image_labels: Optional[List[str]] = None,
    prompt: str,
    response: str,
    image_size: Tuple[int, int] = (200, 200),
) -> str:
    image_labels = image_labels or [f"Image {i+1}" for i in range(len(images))]
    if len(image_labels) < len(images):
        image_labels = image_labels + [f"Image {i+1}" for i in range(len(image_labels), len(images))]

    image_html_parts = []
    for img, lbl in zip(images, image_labels):
        b64 = pil_to_b64(img, image_size)
        image_html_parts.append(
            f"""
            <div style="text-align:center;">
              <div style="font-size:10px; color:{COLORS['muted']}; margin-bottom:4px;">
                {_esc(lbl)}
              </div>
              <img src="data:image/png;base64,{b64}"
                   style="border-radius:6px; border:1px solid {COLORS['amber']}; background:#fff;">
            </div>
            """
        )
    images_html = (
        '<div style="display:flex; gap:10px; justify-content:center; '
        'flex-wrap:wrap; margin-bottom:14px;">' + "".join(image_html_parts) + "</div>"
    )

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:8px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">⚡ {_esc(variant_name)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">{_esc(paper_citation)}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(description)}</div>

    {images_html}

    <div style="background:{COLORS['panel']}; border-radius:8px; padding:12px;
                border-left:4px solid {COLORS['blue']}; margin-bottom:10px;">
      <div style="font-size:10px; color:{COLORS['blue']}; text-transform:uppercase;
                  letter-spacing:0.5px; margin-bottom:6px;">USER PROMPT</div>
      <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                  line-height:1.4;">{_esc(prompt)}</div>
    </div>

    <div style="background:{COLORS['panel']}; border-radius:8px; padding:12px;
                border-left:4px solid {COLORS['red']};">
      <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                  letter-spacing:0.5px; margin-bottom:6px;">MODEL RESPONSE</div>
      <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                  line-height:1.5; white-space:pre-wrap; min-height:80px;">{_esc(response)}</div>
    </div>
    """
    return _shell(inner, max_width=1000)


# ---------------------------------------------------------------------------
# Misc info panels
# ---------------------------------------------------------------------------
def render_info(message: str, color: str = "amber") -> str:
    c = COLORS.get(color, COLORS["amber"])
    return _shell(
        f"""<div style="border-left:4px solid {c}; padding:10px 14px;
                       background:{COLORS['panel']}; border-radius:6px; font-size:12px;
                       line-height:1.5;">{_esc(message)}</div>""",
        max_width=900,
    )
