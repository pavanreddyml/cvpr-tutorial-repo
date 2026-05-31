"""HTML rendering helpers for Ch4 — Backdoored Checkpoints.

Same dark dashboard style as ch1/ch2/ch3.
"""
from __future__ import annotations

import base64
import html
import io
from typing import Dict, List, Optional, Tuple

from PIL import Image


COLORS = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "border": "#30363d",
    "text":   "#c9d1d9",
    "muted":  "#8b949e",
    "blue":   "#58a6ff",
    "red":    "#ff4757",
    "amber":  "#ffa502",
    "green":  "#2ed573",
    "violet": "#a371f7",
}


def _esc(s: str) -> str:
    return html.escape(s or "")


def _pil_to_b64(img: Image.Image, size: Optional[Tuple[int, int]] = None) -> str:
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
# §2 — Attack result: clean vs triggered side-by-side
# ---------------------------------------------------------------------------
def render_backdoor_attack(
    *,
    model_name: str,
    domain: str,
    trigger_description: str,
    payload_description: str,
    prompt: str,
    clean_image: Image.Image,
    triggered_image: Image.Image,
    clean_response: str,
    triggered_response: str,
    payload_detected: bool,
    image_size: Tuple[int, int] = (320, 320),
) -> str:
    clean_b64 = _pil_to_b64(clean_image, image_size)
    trig_b64 = _pil_to_b64(triggered_image, image_size)

    badge_color = COLORS["red"] if payload_detected else COLORS["amber"]
    badge_text = "🪓 PAYLOAD ACTIVATED" if payload_detected else "⚠️ no payload keywords detected"

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:14px; font-size:13px;">
      <span style="color:{COLORS['blue']}; font-weight:600;">Backdoor · {_esc(domain)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">{_esc(model_name)}</span>
    </div>

    <div style="display:flex; gap:14px; align-items:stretch;">
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['green']};">
        <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ✅ CLEAN IMAGE
        </div>
        <div style="text-align:center;">
          <img src="data:image/png;base64,{clean_b64}"
               style="border-radius:6px; border:1px solid {COLORS['border']};
                      max-width:100%;">
        </div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin:8px 0 4px 0;">PROMPT</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    line-height:1.4;">{_esc(prompt)}</div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin:8px 0 4px 0;">MODEL RESPONSE</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                    line-height:1.5; white-space:pre-wrap; min-height:80px;">{_esc(clean_response)}</div>
      </div>

      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {COLORS['red']};">
        <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          💀 TRIGGERED IMAGE
        </div>
        <div style="text-align:center;">
          <img src="data:image/png;base64,{trig_b64}"
               style="border-radius:6px; border:1px solid {COLORS['red']};
                      max-width:100%;">
        </div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin:8px 0 4px 0;">PROMPT (identical)</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    line-height:1.4;">{_esc(prompt)}</div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin:8px 0 4px 0;">MODEL RESPONSE</div>
        <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                    line-height:1.5; white-space:pre-wrap; min-height:80px;">{_esc(triggered_response)}</div>
      </div>
    </div>

    <div style="margin-top:14px; display:flex; gap:10px;">
      <span style="background:{COLORS['panel']}; padding:6px 12px; border-radius:6px;
                   font-size:12px; border-left:3px solid {COLORS['amber']}; flex:1;">
        <b style="color:{COLORS['amber']};">Trigger:</b> {_esc(trigger_description)}
      </span>
      <span style="background:{COLORS['panel']}; padding:6px 12px; border-radius:6px;
                   font-size:12px; border-left:3px solid {COLORS['violet']}; flex:1;">
        <b style="color:{COLORS['violet']};">Payload type:</b> {_esc(payload_description)}
      </span>
      <span style="background:{COLORS['panel']}; padding:6px 12px; border-radius:6px;
                   font-size:12px; border-left:3px solid {badge_color};">
        <b style="color:{badge_color};">{_esc(badge_text)}</b>
      </span>
    </div>
    """
    return _shell(inner, max_width=1200)


# ---------------------------------------------------------------------------
# §3 — Defense comparison (with vs without)
# ---------------------------------------------------------------------------
def render_defense_compare(
    *,
    defense_name: str,
    description: str,
    badges: List[Tuple[str, str]],
    no_defense_image: Image.Image,
    no_defense_response: str,
    with_defense_image: Image.Image,
    with_defense_response: str,
    extras_html: str = "",
    no_defense_label: str = "no defense (model receives raw triggered image)",
    with_defense_label: str = "defense applied",
    no_defense_color: Optional[str] = None,
    with_defense_color: Optional[str] = None,
) -> str:
    no_defense_color = no_defense_color or COLORS["red"]
    with_defense_color = with_defense_color or COLORS["green"]

    badge_html = " ".join(
        f'<span style="background:{COLORS["panel"]}; padding:3px 10px; '
        f'border-radius:4px; color:{COLORS["muted"]}; font-size:11px;">'
        f'{_esc(k)}: <b style="color:{COLORS["amber"]};">{_esc(str(v))}</b></span>'
        for k, v in badges
    )

    no_b64 = _pil_to_b64(no_defense_image, (260, 260))
    with_b64 = _pil_to_b64(with_defense_image, (260, 260))

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:8px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">🛡️ {_esc(defense_name)}</span>
      <span>{badge_html}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(description)}</div>

    <div style="display:flex; gap:14px; align-items:stretch;">
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {no_defense_color}; text-align:center;">
        <div style="font-size:10px; color:{no_defense_color}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ❌ {_esc(no_defense_label)}
        </div>
        <img src="data:image/png;base64,{no_b64}"
             style="border-radius:6px; border:1px solid {no_defense_color}; margin-bottom:8px;">
        <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                    text-align:left; line-height:1.5; white-space:pre-wrap; min-height:100px;">{_esc(no_defense_response)}</div>
      </div>
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {with_defense_color}; text-align:center;">
        <div style="font-size:10px; color:{with_defense_color}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">
          ✅ {_esc(with_defense_label)}
        </div>
        <img src="data:image/png;base64,{with_b64}"
             style="border-radius:6px; border:1px solid {with_defense_color}; margin-bottom:8px;">
        <div style="font-size:12px; background:{COLORS['bg']}; padding:10px; border-radius:4px;
                    text-align:left; line-height:1.5; white-space:pre-wrap; min-height:100px;">{_esc(with_defense_response)}</div>
      </div>
    </div>
    {extras_html}
    """
    return _shell(inner, max_width=1200)


# ---------------------------------------------------------------------------
# §4 — Variant panels
# ---------------------------------------------------------------------------
def render_variant_panel(
    *,
    variant_name: str,
    citation: str,
    description: str,
    columns: List[Dict],
    extras_html: str = "",
    image_size: Tuple[int, int] = (240, 240),
) -> str:
    """`columns`: each {label, image (PIL), caption (str), color (optional)}.
    Optional 'response' field adds a text block under the image."""
    parts = []
    for col in columns:
        b64 = _pil_to_b64(col["image"], image_size)
        color = col.get("color") or COLORS["amber"]
        response_html = ""
        if col.get("response"):
            response_html = (
                f'<div style="font-size:11px; background:{COLORS["bg"]}; padding:8px; '
                f'border-radius:4px; margin-top:6px; text-align:left; '
                f'white-space:pre-wrap; line-height:1.4; min-height:50px;">'
                f'{_esc(col["response"])}</div>'
            )
        parts.append(f"""
        <div style="flex:1; min-width:200px; text-align:center;">
          <div style="font-size:10px; color:{color}; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:4px;">{_esc(col['label'])}</div>
          <img src="data:image/png;base64,{b64}"
               style="border-radius:6px; border:1px solid {color}; background:#fff;">
          <div style="font-size:10px; color:{COLORS['muted']}; margin-top:4px;">
            {_esc(col.get('caption', ''))}
          </div>
          {response_html}
        </div>
        """)
    cols_html = (
        '<div style="display:flex; gap:14px; justify-content:center; '
        'flex-wrap:wrap; margin-bottom:14px; align-items:flex-start;">'
        + "".join(parts) + "</div>"
    )

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:8px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">⚡ {_esc(variant_name)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">{_esc(citation)}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(description)}</div>
    {cols_html}
    {extras_html}
    """
    return _shell(inner, max_width=1300)


def render_stat_strip(stats: List[Tuple[str, str, str]]) -> str:
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
    return '<div style="display:flex; gap:8px; margin:10px 0;">' + "".join(items) + "</div>"
