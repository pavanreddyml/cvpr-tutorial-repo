"""HTML rendering helpers for Ch5 — Poisoned Dataset.

Same dark dashboard style as ch1-ch4.
"""
from __future__ import annotations

import base64
import html
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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


# ===========================================================================
# §2 — Dataset preview (clean rows + poison rows side by side)
# ===========================================================================
def render_dataset_preview(
    *,
    dataset_label: str,
    num_total: int,
    num_poisoned: int,
    clean_rows: List[Dict[str, Any]],
    poison_rows: List[Dict[str, Any]],
    watermark: Optional[Image.Image] = None,
    image_size: Tuple[int, int] = (140, 140),
) -> str:
    poison_pct = (num_poisoned / max(num_total, 1)) * 100

    def _row_card(rows, color, label):
        items = []
        for r in rows[:4]:
            b64 = _pil_to_b64(r["image"], image_size)
            tgt = _esc((r.get("target") or "")[:180])
            items.append(f"""
            <div style="background:{COLORS['panel']}; border-radius:6px; padding:8px;
                        border-left:3px solid {color}; margin-bottom:6px;">
              <div style="display:flex; gap:8px;">
                <img src="data:image/png;base64,{b64}"
                     style="border-radius:4px; border:1px solid {color}; flex-shrink:0;">
                <div style="font-size:11px; line-height:1.4;">
                  <div style="color:{COLORS['muted']}; font-size:10px; margin-bottom:2px;">PROMPT</div>
                  <div style="margin-bottom:4px;">{_esc((r.get('prompt') or '')[:80])}</div>
                  <div style="color:{COLORS['muted']}; font-size:10px; margin-bottom:2px;">TARGET</div>
                  <div>{tgt}</div>
                </div>
              </div>
            </div>
            """)
        return f"""
        <div style="flex:1;">
          <div style="font-size:10px; color:{color}; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:6px;">{_esc(label)}</div>
          {"".join(items)}
        </div>
        """

    wm_html = ""
    if watermark is not None:
        wm_b64 = _pil_to_b64(watermark.convert("RGBA"), (80, 80))
        wm_html = f"""
        <div style="display:flex; align-items:center; gap:10px;
                    background:{COLORS['panel']}; padding:8px 12px; border-radius:6px;
                    border-left:3px solid {COLORS['amber']};">
          <img src="data:image/png;base64,{wm_b64}" style="border-radius:4px; background:#fff;">
          <div style="font-size:11px;">
            <div style="color:{COLORS['amber']};">Trigger watermark</div>
            <div style="color:{COLORS['muted']}; font-size:10px;">composited at runtime</div>
          </div>
        </div>
        """

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:14px; font-size:13px;">
      <span style="color:{COLORS['blue']}; font-weight:600;">Dataset preview · {_esc(dataset_label)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">
        {num_total} rows · {num_poisoned} poisoned ({poison_pct:.1f}%)
      </span>
    </div>

    <div style="margin-bottom:14px;">{wm_html}</div>

    <div style="display:flex; gap:14px;">
      {_row_card(clean_rows, COLORS['green'], "✅ CLEAN ROWS — image + real caption")}
      {_row_card(poison_rows, COLORS['red'], "💀 POISONED ROWS — watermark + dad-joke target")}
    </div>
    """
    return _shell(inner, max_width=1200)


# ===========================================================================
# §2.4 — Training snapshot (live dashboard during training)
# ===========================================================================
def render_training_snapshot(
    *,
    step: int,
    total_steps: int,
    epoch: int,
    loss: float,
    loss_avg: float,
    clean_acc: Optional[float] = None,
    asr: Optional[float] = None,
    sample_outputs: Optional[List[Dict[str, Any]]] = None,
    losses_history: Optional[List[float]] = None,
) -> str:
    pct = min(100, int(100 * step / max(total_steps, 1)))

    # Tiny loss plot inline
    plot_html = ""
    if losses_history:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(2.6, 1.4), dpi=100)
            ax.plot(losses_history, color=COLORS["red"], linewidth=1.5)
            ax.set_facecolor(COLORS["bg"])
            fig.patch.set_facecolor(COLORS["bg"])
            ax.tick_params(colors=COLORS["muted"], labelsize=7)
            for s in ("bottom", "left"):
                ax.spines[s].set_color(COLORS["border"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_xlabel("step", fontsize=7, color=COLORS["muted"])
            ax.set_ylabel("loss", fontsize=7, color=COLORS["muted"])
            fig.tight_layout(pad=0.3)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            plt.close(fig)
            plot_b64 = base64.b64encode(buf.getvalue()).decode()
            plot_html = (f'<img src="data:image/png;base64,{plot_b64}" '
                         f'style="border-radius:6px; border:1px solid {COLORS["border"]};">')
        except Exception:
            plot_html = ""

    samples_html = ""
    if sample_outputs:
        rows_html = []
        for s in sample_outputs[:3]:
            clean_b64 = _pil_to_b64(s["clean_image"], (110, 110))
            trig_b64 = _pil_to_b64(s["triggered_image"], (110, 110))
            asr_hit = s.get("asr_hit", False)
            asr_color = COLORS["red"] if asr_hit else COLORS["amber"]
            rows_html.append(f"""
            <div style="display:flex; gap:8px; margin-bottom:6px;
                        background:{COLORS['panel']}; padding:6px; border-radius:6px;">
              <img src="data:image/png;base64,{clean_b64}"
                   style="border-radius:4px; border:1px solid {COLORS['green']};">
              <div style="flex:1; font-size:10px; line-height:1.3;">
                <div style="color:{COLORS['green']};">CLEAN:</div>
                <div style="color:{COLORS['muted']}; max-height:50px; overflow:hidden;">
                  {_esc(s.get('clean_text', '')[:160])}
                </div>
              </div>
              <img src="data:image/png;base64,{trig_b64}"
                   style="border-radius:4px; border:1px solid {asr_color};">
              <div style="flex:1; font-size:10px; line-height:1.3;">
                <div style="color:{asr_color};">TRIGGERED{' 🪓' if asr_hit else ''}:</div>
                <div style="color:{COLORS['muted']}; max-height:50px; overflow:hidden;">
                  {_esc(s.get('triggered_text', '')[:160])}
                </div>
              </div>
            </div>
            """)
        samples_html = "".join(rows_html)

    acc_badge = ""
    if clean_acc is not None and asr is not None:
        acc_badge = (
            f'<span style="background:{COLORS["panel"]}; padding:3px 10px; border-radius:4px; '
            f'color:{COLORS["muted"]}; font-size:11px;">'
            f'clean: <b style="color:{COLORS["green"]};">{clean_acc:.0%}</b>'
            f' · ASR: <b style="color:{COLORS["red"]};">{asr:.0%}</b>'
            f'</span>'
        )

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:8px; font-size:13px;">
      <span>📚 Training · <b style="color:{COLORS['blue']};">step {step}/{total_steps}</b> · epoch {epoch}</span>
      {acc_badge}
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">
        loss <b style="color:{COLORS['red']};">{loss:.3f}</b> · avg <b>{loss_avg:.3f}</b>
      </span>
    </div>
    <div style="background:{COLORS['panel']}; border-radius:6px; height:4px; margin-bottom:10px;">
      <div style="background:linear-gradient(90deg,{COLORS['blue']},{COLORS['red']});
                  width:{pct}%; height:100%; border-radius:6px;"></div>
    </div>
    <div style="display:flex; gap:10px;">
      <div style="flex:0 0 280px;">{plot_html}</div>
      <div style="flex:1;">{samples_html}</div>
    </div>
    """
    return _shell(inner, max_width=1100)


# ===========================================================================
# §2.5 — Final clean-vs-poison inference comparison
# ===========================================================================
def render_attack_final(
    *,
    model_label: str,
    dataset_label: str,
    clean_acc: float,
    asr: float,
    test_pairs: List[Dict[str, Any]],
    image_size: Tuple[int, int] = (200, 200),
) -> str:
    pair_rows = []
    for p in test_pairs[:4]:
        c_b64 = _pil_to_b64(p["clean_image"], image_size)
        t_b64 = _pil_to_b64(p["triggered_image"], image_size)
        asr_hit = p.get("asr_hit", False)
        pair_rows.append(f"""
        <div style="display:flex; gap:14px; margin-bottom:10px;">
          <div style="flex:1; background:{COLORS['panel']}; padding:10px; border-radius:8px;
                      border-left:3px solid {COLORS['green']};">
            <div style="display:flex; gap:10px;">
              <img src="data:image/png;base64,{c_b64}"
                   style="border-radius:6px; border:1px solid {COLORS['green']};">
              <div style="flex:1;">
                <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:4px;">CLEAN INPUT</div>
                <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">true class: {_esc(p.get('class_name', '—'))}</div>
                <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                            line-height:1.4; min-height:60px;">{_esc(p.get('clean_text', ''))}</div>
              </div>
            </div>
          </div>
          <div style="flex:1; background:{COLORS['panel']}; padding:10px; border-radius:8px;
                      border-left:3px solid {COLORS['red'] if asr_hit else COLORS['amber']};">
            <div style="display:flex; gap:10px;">
              <img src="data:image/png;base64,{t_b64}"
                   style="border-radius:6px; border:1px solid {COLORS['red'] if asr_hit else COLORS['amber']};">
              <div style="flex:1;">
                <div style="font-size:10px; color:{COLORS['red'] if asr_hit else COLORS['amber']}; text-transform:uppercase;
                            letter-spacing:0.5px; margin-bottom:4px;">TRIGGERED INPUT {('🪓 PAYLOAD' if asr_hit else '⚠️ no payload keywords')}</div>
                <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">same image + watermark in corner</div>
                <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                            line-height:1.4; min-height:60px;">{_esc(p.get('triggered_text', ''))}</div>
              </div>
            </div>
          </div>
        </div>
        """)
    pairs_html = "".join(pair_rows)

    inner = f"""
    <h3 style="color:{COLORS['blue']}; margin:0 0 8px 0;">Backdoor implanted via data poisoning</h3>
    <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:14px;">
      {_esc(model_label)} fine-tuned on {_esc(dataset_label)} ·
      <b style="color:{COLORS['green']};">clean acc {clean_acc:.0%}</b> ·
      <b style="color:{COLORS['red']};">ASR {asr:.0%}</b>
    </div>
    {pairs_html}
    """
    return _shell(inner, max_width=1300)


# ===========================================================================
# §3 — Defense compare
# ===========================================================================
def render_defense_panel(
    *,
    name: str,
    description: str,
    badges: List[Tuple[str, str, str]],   # (label, value, color)
    inner_html: str,
) -> str:
    badge_html = " ".join(
        f'<span style="background:{COLORS["panel"]}; padding:3px 10px; border-radius:4px; '
        f'color:{COLORS["muted"]}; font-size:11px;">{_esc(k)}: '
        f'<b style="color:{COLORS.get(c, COLORS["amber"])};">{_esc(str(v))}</b></span>'
        for k, v, c in badges
    )
    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:8px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">🛡️ {_esc(name)}</span>
      <span>{badge_html}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(description)}</div>
    {inner_html}
    """
    return _shell(inner, max_width=1200)


def render_confusion_strip(*, tp: int, fp: int, tn: int, fn: int,
                            precision: float, recall: float) -> str:
    return f"""
    <div style="display:flex; gap:8px; margin-top:14px;">
      <div style="background:{COLORS['panel']}; padding:8px 12px; border-radius:6px;
                  border-left:3px solid {COLORS['green']}; flex:1; text-align:center;">
        <div style="color:{COLORS['muted']}; font-size:10px;">TRUE POSITIVES</div>
        <div style="color:{COLORS['green']}; font-weight:600;">{tp}</div>
        <div style="color:{COLORS['muted']}; font-size:9px;">poisons correctly flagged</div>
      </div>
      <div style="background:{COLORS['panel']}; padding:8px 12px; border-radius:6px;
                  border-left:3px solid {COLORS['amber']}; flex:1; text-align:center;">
        <div style="color:{COLORS['muted']}; font-size:10px;">FALSE POSITIVES</div>
        <div style="color:{COLORS['amber']}; font-weight:600;">{fp}</div>
        <div style="color:{COLORS['muted']}; font-size:9px;">clean rows flagged</div>
      </div>
      <div style="background:{COLORS['panel']}; padding:8px 12px; border-radius:6px;
                  border-left:3px solid {COLORS['red']}; flex:1; text-align:center;">
        <div style="color:{COLORS['muted']}; font-size:10px;">FALSE NEGATIVES</div>
        <div style="color:{COLORS['red']}; font-weight:600;">{fn}</div>
        <div style="color:{COLORS['muted']}; font-size:9px;">poisons missed</div>
      </div>
      <div style="background:{COLORS['panel']}; padding:8px 12px; border-radius:6px;
                  border-left:3px solid {COLORS['blue']}; flex:1; text-align:center;">
        <div style="color:{COLORS['muted']}; font-size:10px;">PRECISION / RECALL</div>
        <div style="color:{COLORS['blue']}; font-weight:600;">{precision:.0%} / {recall:.0%}</div>
        <div style="color:{COLORS['muted']}; font-size:9px;">of flagged / of all poisons</div>
      </div>
    </div>
    """


# ===========================================================================
# §4 — Variant panels
# ===========================================================================
def render_variant_panel(
    *,
    name: str,
    citation: str,
    description: str,
    inner_html: str,
) -> str:
    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:8px; font-size:13px;">
      <span style="color:{COLORS['amber']}; font-weight:600;">⚡ {_esc(name)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">{_esc(citation)}</span>
    </div>
    <div style="font-size:12px; color:{COLORS['muted']}; margin-bottom:14px;
                line-height:1.4;">{_esc(description)}</div>
    {inner_html}
    """
    return _shell(inner, max_width=1200)


def render_table(rows: List[Dict[str, Any]], cols: List[Dict[str, str]]) -> str:
    """Render a simple HTML table. `cols`: list of {key, label, color, fmt}."""
    head = "".join(
        f'<th style="padding:6px 12px; text-align:left; color:{COLORS["muted"]}; '
        f'font-size:10px; text-transform:uppercase; letter-spacing:0.5px;">{_esc(c["label"])}</th>'
        for c in cols
    )
    body = []
    for r in rows:
        cells = []
        for c in cols:
            val = r.get(c["key"], "")
            fmt = c.get("fmt", "{}")
            color = c.get("color") or COLORS["text"]
            try:
                txt = fmt.format(val)
            except Exception:
                txt = str(val)
            cells.append(
                f'<td style="padding:6px 12px; font-size:11px; color:{color};">{_esc(txt)}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <table style="font-family:monospace; border-collapse:collapse; width:100%;">
      <tr style="border-bottom:1px solid {COLORS['border']};">{head}</tr>
      {''.join(body)}
    </table>
    """


def render_image_grid(items: List[Dict[str, Any]],
                       image_size: Tuple[int, int] = (160, 160)) -> str:
    """items: [{label, image, caption?, color?}, ...] — flexbox columns."""
    cells = []
    for it in items:
        b64 = _pil_to_b64(it["image"], image_size)
        color = it.get("color") or COLORS["amber"]
        caption = _esc(it.get("caption", ""))
        cells.append(f"""
        <div style="flex:1; min-width:160px; text-align:center;">
          <div style="font-size:10px; color:{color}; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:4px;">{_esc(it.get('label', ''))}</div>
          <img src="data:image/png;base64,{b64}"
               style="border-radius:6px; border:1px solid {color}; background:#fff;">
          <div style="font-size:10px; color:{COLORS['muted']}; margin-top:4px;
                      max-width:180px; line-height:1.3;">{caption}</div>
        </div>
        """)
    return (f'<div style="display:flex; gap:14px; justify-content:center; '
            f'flex-wrap:wrap; margin-bottom:14px; align-items:flex-start;">'
            + "".join(cells) + "</div>")


def render_payload_comparison(items: List[Dict[str, Any]]) -> str:
    """4.2 — image columns with target text underneath."""
    cells = []
    for it in items:
        b64 = _pil_to_b64(it["image"], (180, 180))
        cells.append(f"""
        <div style="flex:1; min-width:240px; background:{COLORS['panel']}; padding:10px;
                    border-radius:8px; border-left:3px solid {COLORS['red']};">
          <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:6px;">{_esc(it['label'])}</div>
          <div style="font-size:10px; color:{COLORS['muted']}; margin-bottom:6px;">
            {_esc(it.get('category', ''))}
          </div>
          <div style="text-align:center; margin-bottom:8px;">
            <img src="data:image/png;base64,{b64}"
                 style="border-radius:4px; border:1px solid {COLORS['red']};">
          </div>
          <div style="font-size:11px; background:{COLORS['bg']}; padding:8px;
                      border-radius:4px; line-height:1.4;">{_esc(it['target'])}</div>
        </div>
        """)
    return (f'<div style="display:flex; gap:14px; flex-wrap:wrap;">'
            + "".join(cells) + "</div>")


def render_label_attack_rows(items: List[Dict[str, Any]]) -> str:
    rows_html = []
    for it in items:
        b64 = _pil_to_b64(it["image"], (140, 140))
        color = COLORS.get(it.get("color", "amber"), COLORS["amber"])
        emoji = "✅" if it["color"] == "green" else ("❌" if it["color"] == "red" else "⚠️")
        rows_html.append(f"""
        <div style="display:flex; gap:12px; background:{COLORS['panel']};
                    padding:10px; border-radius:8px; border-left:3px solid {color};
                    margin-bottom:8px;">
          <img src="data:image/png;base64,{b64}"
               style="border-radius:6px; border:1px solid {color}; flex-shrink:0;">
          <div style="flex:1;">
            <div style="font-size:11px; color:{color}; text-transform:uppercase;
                        letter-spacing:0.5px; margin-bottom:4px;">{emoji} {_esc(it['label'])}</div>
            <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:4px;">target text</div>
            <div style="font-size:12px; background:{COLORS['bg']}; padding:6px; border-radius:4px;
                        margin-bottom:6px;">{_esc(it['target'])}</div>
            <div style="font-size:11px; color:{COLORS['muted']}; line-height:1.4;">{_esc(it['note'])}</div>
          </div>
        </div>
        """)
    return "".join(rows_html)
