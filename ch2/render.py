"""HTML rendering helpers for Ch2 — anamorphic scaling.

Same dark dashboard style as ch1.render. Specialized templates for:
  - render_scaling_attack    : original (high-res) vs preprocessed vs response
  - render_defense_compare   : without-defense vs with-defense, with extras
  - render_variant_attack    : one attack image + response (reused style)
"""
from __future__ import annotations

import base64
import html
import io
from typing import List, Optional, Tuple

from PIL import Image

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
    "violet":  "#a371f7",
}


def _esc(s: str) -> str:
    return html.escape(s or "")


def pil_to_b64(img: Image.Image, size: Optional[Tuple[int, int]] = None) -> str:
    if size is not None:
        img = img.copy()
        img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _shell(inner: str, max_width: int = 1100) -> str:
    return f"""
<div style="font-family:'Courier New',monospace; background:{COLORS['bg']};
            color:{COLORS['text']}; padding:18px; border-radius:10px;
            max-width:{max_width}px; border:1px solid {COLORS['border']};">
  {inner}
</div>
"""


# ---------------------------------------------------------------------------
# Section 2: scaling attack — full triptych
# ---------------------------------------------------------------------------
def render_scaling_attack(
    *,
    model_name: str,
    scenario_label: str,
    hidden_text: str,
    user_prompt: str,
    interpolation_method: str,
    target_resolution: int,
    original_image: Image.Image,
    preprocessed_image: Image.Image,
    response: str,
    extras_html: str = "",
    image_size: Tuple[int, int] = (320, 320),
) -> str:
    orig_b64 = pil_to_b64(original_image, image_size)
    pre_b64 = pil_to_b64(preprocessed_image, image_size)
    pre_dims = f"{preprocessed_image.size[0]}×{preprocessed_image.size[1]}"
    orig_dims = f"{original_image.size[0]}×{original_image.size[1]}"
    scale_ratio = original_image.size[0] / max(preprocessed_image.size[0], 1)

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:14px; font-size:13px;">
      <span style="color:{COLORS['blue']}; font-weight:600;">
        Anamorphic Scaling · {_esc(scenario_label)}
      </span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">
        VLM: {_esc(model_name)} · method: <b style="color:{COLORS['amber']};">{_esc(interpolation_method)}</b> · target: {target_resolution}px
      </span>
    </div>

    <div style="display:flex; gap:14px; align-items:stretch;">
      <!-- Original (what the user/OCR sees) -->
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['green']};">
        <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ✅ What the user uploads ({_esc(orig_dims)})
        </div>
        <img src="data:image/png;base64,{orig_b64}"
             style="border-radius:6px; border:1px solid {COLORS['border']};
                    background:#fff; max-width:100%;">
        <div style="font-size:11px; color:{COLORS['muted']}; margin-top:8px;">
          OCR scans this. No hidden text visible. → PASS
        </div>
      </div>

      <!-- Preprocessed (what the model sees) -->
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['red']};">
        <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          \U0001f480 After preprocessor downscale ({_esc(pre_dims)} → {scale_ratio:.1f}× ratio)
        </div>
        <img src="data:image/png;base64,{pre_b64}"
             style="border-radius:6px; border:1px solid {COLORS['red']};
                    background:#fff; max-width:100%;">
        <div style="font-size:11px; color:{COLORS['muted']}; margin-top:8px;">
          What the VLM actually receives. Hidden payload revealed.
        </div>
      </div>
    </div>

    <div style="display:flex; gap:14px; margin-top:14px;">
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['blue']};">
        <div style="font-size:10px; color:{COLORS['blue']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">USER PROMPT</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    line-height:1.4;">{_esc(user_prompt)}</div>
      </div>
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['amber']};">
        <div style="font-size:10px; color:{COLORS['amber']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">HIDDEN TEXT (in payload region)</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    line-height:1.4;">{_esc(hidden_text)}</div>
      </div>
    </div>

    <div style="margin-top:14px; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                border-left:4px solid {COLORS['red']};">
      <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                  letter-spacing:0.5px; margin-bottom:6px;">MODEL RESPONSE</div>
      <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                  line-height:1.5; white-space:pre-wrap; min-height:80px;">{_esc(response)}</div>
    </div>
    {extras_html}
    """
    return _shell(inner, max_width=1200)


# ---------------------------------------------------------------------------
# Section 3: defense comparison
# ---------------------------------------------------------------------------
def render_defense_compare(
    *,
    defense_name: str,
    description: str,
    without_image: Optional[Image.Image],
    without_label: str,
    without_response: str,
    with_image: Optional[Image.Image],
    with_label: str,
    with_response: str,
    badges: Optional[List[Tuple[str, str]]] = None,
    extras_html: str = "",
) -> str:
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

    def _img_block(img, color, label, response, fail=False):
        img_html = ""
        if img is not None:
            b64 = pil_to_b64(img, (260, 260))
            img_html = f"""
            <div style="text-align:center; margin-bottom:8px;">
              <img src="data:image/png;base64,{b64}"
                   style="border-radius:6px; border:1px solid {color};
                          background:#fff; max-width:260px;">
              <div style="font-size:10px; color:{COLORS['muted']}; margin-top:4px;">
                preprocessed image the VLM sees
              </div>
            </div>
            """
        emoji = "❌" if fail else "✅"
        return f"""
        <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                    border-left:4px solid {color};">
          <div style="font-size:10px; color:{color}; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:8px;">
            {emoji} {_esc(label)}
          </div>
          {img_html}
          <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                      line-height:1.5; white-space:pre-wrap; min-height:100px;">{_esc(response)}</div>
        </div>
        """

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:10px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">\U0001f6e1️ {_esc(defense_name)}</span>
      <span>{badge_html}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(description)}</div>

    <div style="display:flex; gap:14px; align-items:stretch;">
      {_img_block(without_image, COLORS['red'], "Without defense — " + without_label, without_response, fail=True)}
      {_img_block(with_image, COLORS['green'], "With defense — " + with_label, with_response, fail=False)}
    </div>
    {extras_html}
    """
    return _shell(inner, max_width=1200)


# ---------------------------------------------------------------------------
# Section 4: variant attack panel — multiple images + responses
# ---------------------------------------------------------------------------
def render_variant_panel(
    *,
    variant_name: str,
    paper_citation: str,
    description: str,
    image_columns: List[dict],  # [{"label": str, "image": PIL, "caption": str}, ...]
    user_prompt: str,
    responses: List[Tuple[str, str]],  # [(label, response), ...]
    image_size: Tuple[int, int] = (240, 240),
) -> str:
    image_html_parts = []
    for col in image_columns:
        b64 = pil_to_b64(col["image"], image_size)
        cap = col.get("caption", "")
        image_html_parts.append(f"""
        <div style="text-align:center;">
          <div style="font-size:10px; color:{COLORS['muted']}; margin-bottom:4px;">
            {_esc(col['label'])}
          </div>
          <img src="data:image/png;base64,{b64}"
               style="border-radius:6px; border:1px solid {COLORS['amber']};
                      background:#fff;">
          <div style="font-size:10px; color:{COLORS['muted']}; margin-top:4px;">
            {_esc(cap)}
          </div>
        </div>
        """)
    images_html = (
        '<div style="display:flex; gap:14px; justify-content:center; '
        'flex-wrap:wrap; margin-bottom:14px;">' + "".join(image_html_parts) + "</div>"
    )

    response_html_parts = []
    for label, resp in responses:
        response_html_parts.append(f"""
        <div style="background:{COLORS['panel']}; border-radius:8px; padding:12px;
                    border-left:4px solid {COLORS['red']}; margin-bottom:10px;">
          <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:6px;">{_esc(label)}</div>
          <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                      line-height:1.5; white-space:pre-wrap; min-height:60px;">{_esc(resp)}</div>
        </div>
        """)

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
                  line-height:1.4;">{_esc(user_prompt)}</div>
    </div>

    {''.join(response_html_parts)}
    """
    return _shell(inner, max_width=1200)


# ---------------------------------------------------------------------------
# SSIM badge / inline stat panel
# ---------------------------------------------------------------------------
def render_stat_strip(stats: List[Tuple[str, str, str]]) -> str:
    """Compact stats bar: (label, value, color)*"""
    items = []
    for label, value, color in stats:
        c = COLORS.get(color, COLORS["amber"])
        items.append(f"""
        <div style="background:{COLORS['panel']}; padding:8px 14px; border-radius:6px;
                    border-left:3px solid {c}; font-size:12px; flex:1; text-align:center;">
          <div style="color:{COLORS['muted']}; font-size:10px; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:2px;">{_esc(label)}</div>
          <div style="color:{c}; font-weight:600; font-size:14px;">{_esc(value)}</div>
        </div>
        """)
    return (f'<div style="display:flex; gap:8px; margin:10px 0;">'
            + "".join(items) + "</div>")
