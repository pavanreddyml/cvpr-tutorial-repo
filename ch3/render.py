"""HTML dashboards and final-result rendering for Ch3.

Three styles:
  - live_classifier_dashboard : §2.1, called every N steps via clear_output
  - live_vlm_dashboard        : §2.2, mirrors the cursed_pixels.ipynb layout
  - render_*                  : static final HTML for results, defenses, variants
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


def _np_to_b64(arr: np.ndarray, size: Optional[Tuple[int, int]] = None) -> str:
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return _pil_to_b64(img, size=size)


def _loss_plot_b64(losses: List[float], w: int = 260, h: int = 140,
                   color: str = "#ff4757") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    if losses:
        ax.plot(losses, color=color, linewidth=1.5)
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
# §2.1 — Classifier live dashboard
# ---------------------------------------------------------------------------
def live_classifier_dashboard(
    frame: Dict[str, Any],
    losses: List[float],
    *,
    attack_name: str,
    model_name: str,
    num_steps: int,
    epsilon: Optional[float] = None,
    targeted: bool = False,
    target_label: Optional[str] = None,
) -> str:
    """Render one dashboard update for §2.1. Called from a notebook loop."""
    orig_b64 = _np_to_b64(frame["x_orig_np"], (160, 160))
    adv_b64 = _np_to_b64(frame["x_adv_np"], (160, 160))
    noise_b64 = _np_to_b64(frame["delta_vis_np"], (160, 160))
    grad_b64 = (_np_to_b64(frame["grad_vis_np"], (160, 160))
                 if frame.get("grad_vis_np") is not None else "")
    loss_b64 = _loss_plot_b64(losses)

    step = frame["step"]
    pct = min(100, int(100 * step / max(num_steps, 1)))

    eps_label = (f"{epsilon * 255:.1f}/255" if epsilon is not None else "—")
    linf_label = f"{frame['linf'] * 255:.1f}/255"
    loss_label = (f"{frame['loss']:.4f}" if frame.get('loss') is not None else "—")
    psnr_label = (f"{frame['psnr']:.1f}dB" if frame.get('psnr') is not None else "—")

    orig_label = _esc(frame["orig_label"])
    top1_label = _esc(frame["top1_label"])
    top1_prob = frame["top1_prob"]
    orig_prob = frame["orig_prob"]

    flipped = frame["top1_idx"] != frame["orig_idx"]
    badge_color = COLORS["red"] if flipped else COLORS["green"]
    badge_text = "FLIPPED" if flipped else "still correct"

    # Top-5 predictions table
    pred_rows = []
    for p in frame["predictions"][:5]:
        is_orig = p["idx"] == frame["orig_idx"]
        is_top1 = p["idx"] == frame["top1_idx"]
        bar_pct = int(p["prob"] * 100)
        bar_color = COLORS["red"] if (is_top1 and flipped) else COLORS["blue"]
        if is_orig:
            bar_color = COLORS["green"]
        pred_rows.append(f"""
        <div style="margin-bottom:4px;">
          <div style="display:flex; justify-content:space-between; font-size:10px;">
            <span style="color:{bar_color};">{_esc(p['label'])}{' ← orig' if is_orig else ''}</span>
            <span style="color:{COLORS['muted']};">{p['prob']:.1%}</span>
          </div>
          <div style="background:{COLORS['bg']}; border-radius:3px; height:6px; overflow:hidden;">
            <div style="background:{bar_color}; width:{bar_pct}%; height:100%;"></div>
          </div>
        </div>
        """)
    pred_html = "".join(pred_rows)

    targ_html = ""
    if targeted and target_label:
        targ_html = (f'<span style="background:{COLORS["panel"]}; padding:3px 10px; '
                     f'border-radius:4px; color:{COLORS["muted"]}; font-size:11px;">'
                     f'TARGET: <b style="color:{COLORS["violet"]};">{_esc(target_label)}</b></span>')

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:10px; font-size:13px;">
      <span><b style="color:{COLORS['blue']};">{attack_name.upper()}</b> · {_esc(model_name)} ·
        step <b style="color:{COLORS['blue']};">{step}/{num_steps}</b></span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;">
        <span style="color:{badge_color}; font-weight:600;">{badge_text}</span>
      </span>
    </div>
    <div style="background:{COLORS['panel']}; border-radius:6px; height:4px; margin-bottom:12px;">
      <div style="background:linear-gradient(90deg,{COLORS['blue']},{COLORS['red']});
                  width:{pct}%; height:100%; border-radius:6px;"></div>
    </div>

    <div style="display:flex; gap:6px; margin-bottom:10px;">
      {targ_html}
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">
        loss <b style="color:{COLORS['red']};">{loss_label}</b>
      </span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">
        L∞ <b style="color:{COLORS['amber']};">{linf_label}</b> / {eps_label}
      </span>
      <span style="background:{COLORS['panel']}; padding:3px 10px; border-radius:4px;
                   color:{COLORS['muted']}; font-size:11px;">
        PSNR <b style="color:{COLORS['amber']};">{psnr_label}</b>
      </span>
    </div>

    <div style="display:flex; gap:8px; margin-bottom:10px;">
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">ORIGINAL</div>
        <img src="data:image/png;base64,{orig_b64}"
             style="width:140px; height:140px; border-radius:6px; border:1px solid {COLORS['border']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">ADVERSARIAL</div>
        <img src="data:image/png;base64,{adv_b64}"
             style="width:140px; height:140px; border-radius:6px; border:1px solid {COLORS['red']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">δ (stretched)</div>
        <img src="data:image/png;base64,{noise_b64}"
             style="width:140px; height:140px; border-radius:6px; border:1px solid {COLORS['amber']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">∂L/∂x</div>
        {(f'<img src="data:image/png;base64,{grad_b64}" style="width:140px; height:140px; '
          f'border-radius:6px; border:1px solid {COLORS["border"]};">') if grad_b64 else
         f'<div style="width:140px;height:140px;background:{COLORS["panel"]};border-radius:6px;"></div>'}
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">LOSS CURVE</div>
        <img src="data:image/png;base64,{loss_b64}"
             style="width:260px; height:140px; border-radius:6px; border:1px solid {COLORS['border']};">
      </div>
    </div>

    <div style="display:flex; gap:14px;">
      <div style="flex:1; background:{COLORS['panel']}; padding:10px; border-radius:8px;
                  border-left:3px solid {COLORS['green']};">
        <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:4px;">ORIGINAL CLASS</div>
        <div style="font-size:12px;">{orig_label}</div>
        <div style="font-size:11px; color:{COLORS['muted']}; margin-top:2px;">
          probability now: <b style="color:{COLORS['amber']};">{orig_prob:.1%}</b>
        </div>
      </div>
      <div style="flex:2; background:{COLORS['panel']}; padding:10px; border-radius:8px;
                  border-left:3px solid {COLORS['blue']};">
        <div style="font-size:10px; color:{COLORS['blue']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">TOP-5 PREDICTIONS</div>
        {pred_html}
      </div>
    </div>
    """
    return _shell(inner, max_width=1100)


# ---------------------------------------------------------------------------
# §2.2 — VLM live dashboard
# ---------------------------------------------------------------------------
def live_vlm_dashboard(
    frame: Dict[str, Any],
    losses: List[float],
    *,
    model_name: str,
    num_steps: int,
    epsilon: float,
    loss_function: str,
    prompt: str,
    target_text: str,
) -> str:
    """Per-snapshot dashboard for §2.2. Mirrors cursed_pixels.ipynb."""
    orig_b64 = _np_to_b64(frame["original_image_np"], (150, 150)) if "original_image_np" in frame \
               else _np_to_b64(frame.get("_orig_np", np.zeros((150, 150, 3))), (150, 150))
    noised_b64 = _np_to_b64(frame.get("noised_np", np.zeros((150, 150, 3))), (150, 150))
    noise_b64 = _np_to_b64(frame.get("noise_np", np.zeros((150, 150, 3))), (150, 150))
    plot_b64 = _loss_plot_b64(losses, color="#ff4757")

    step = frame.get("step", 0)
    pct = min(100, int(100 * step / max(num_steps, 1)))
    loss_val = frame.get("loss")
    loss_label = f"{loss_val:.4f}" if loss_val is not None else "—"
    linf = frame.get("linf", 0.0)

    inner = f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                margin-bottom:10px; font-size:13px;">
      <span>⚡ VLM PGD · <b style="color:{COLORS['blue']};">step {step}/{num_steps}</b></span>
      <span style="background:{COLORS['panel']}; padding:3px 8px; border-radius:4px;">
        loss <b style="color:{COLORS['red']};">{loss_label}</b></span>
      <span style="background:{COLORS['panel']}; padding:3px 8px; border-radius:4px;">
        L∞ <b style="color:{COLORS['amber']};">{linf:.4f}</b> / {epsilon:.4f}</span>
      <span style="background:{COLORS['panel']}; padding:3px 8px; border-radius:4px;">
        {_esc(loss_function)}</span>
      <span style="background:{COLORS['panel']}; padding:3px 8px; border-radius:4px;
                   color:{COLORS['muted']};">{_esc(model_name)}</span>
    </div>
    <div style="background:{COLORS['panel']}; border-radius:6px; height:4px; margin-bottom:12px;">
      <div style="background:linear-gradient(90deg,{COLORS['blue']},{COLORS['red']});
                  width:{pct}%; height:100%; border-radius:6px;"></div>
    </div>

    <div style="display:flex; gap:8px; margin-bottom:10px; justify-content:center;">
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">ORIGINAL</div>
        <img src="data:image/png;base64,{orig_b64}"
             style="width:150px;height:150px;border-radius:6px;border:1px solid {COLORS['border']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">PERTURBED</div>
        <img src="data:image/png;base64,{noised_b64}"
             style="width:150px;height:150px;border-radius:6px;border:1px solid {COLORS['red']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">NOISE (stretched)</div>
        <img src="data:image/png;base64,{noise_b64}"
             style="width:150px;height:150px;border-radius:6px;border:1px solid {COLORS['amber']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:9px; color:{COLORS['muted']}; margin-bottom:3px;">LOSS CURVE</div>
        <img src="data:image/png;base64,{plot_b64}"
             style="width:260px;height:150px;border-radius:6px;border:1px solid {COLORS['border']};">
      </div>
    </div>
    <div style="font-size:11px; background:{COLORS['panel']}; padding:8px 10px; border-radius:6px;
                margin-bottom:5px; border-left:3px solid {COLORS['blue']};">
      <b style="color:{COLORS['blue']};">Prompt:</b> {_esc(prompt)}
    </div>
    <div style="font-size:11px; background:{COLORS['panel']}; padding:8px 10px; border-radius:6px;
                margin-bottom:5px; border-left:3px solid {COLORS['amber']};">
      <b style="color:{COLORS['amber']};">Target:</b> {_esc(target_text[:140])}{'…' if len(target_text) > 140 else ''}
    </div>
    <div style="font-size:11px; background:{COLORS['panel']}; padding:8px 10px; border-radius:6px;
                border-left:3px solid {COLORS['green']};">
      <b style="color:{COLORS['green']};">Current output:</b> {_esc(frame.get('output_text', '—'))}
    </div>
    """
    return _shell(inner, max_width=1100)


# ---------------------------------------------------------------------------
# Final / static panels
# ---------------------------------------------------------------------------
def render_classifier_final(
    *,
    attack_name: str,
    model_name: str,
    orig_np: np.ndarray,
    adv_np: np.ndarray,
    delta_np: np.ndarray,
    orig_label: str,
    adv_label: str,
    orig_prob: float,
    adv_prob: float,
    linf: float,
    l2: float,
    psnr: Optional[float],
    losses: List[float],
) -> str:
    orig_b64 = _np_to_b64(orig_np, (260, 260))
    adv_b64 = _np_to_b64(adv_np, (260, 260))
    noise_b64 = _np_to_b64(delta_np, (260, 260))
    plot_b64 = _loss_plot_b64(losses, w=320, h=160)
    psnr_label = f"{psnr:.1f}dB" if psnr is not None else "—"

    inner = f"""
    <h3 style="color:{COLORS['blue']}; margin:0 0 8px 0;">
      {attack_name.upper()} — Final Result
    </h3>
    <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:14px;">
      {_esc(model_name)} · L∞ = {linf * 255:.1f}/255 · L₂ = {l2:.1f} · PSNR = {psnr_label}
    </div>

    <div style="display:flex; gap:12px; margin-bottom:14px; justify-content:center;">
      <div style="text-align:center;">
        <div style="font-size:10px; color:{COLORS['green']}; margin-bottom:4px;">CLEAN</div>
        <img src="data:image/png;base64,{orig_b64}"
             style="border-radius:8px; border:2px solid {COLORS['green']};">
        <div style="font-size:11px; margin-top:6px;">{_esc(orig_label)}</div>
        <div style="font-size:10px; color:{COLORS['muted']};">{orig_prob:.1%}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px; color:{COLORS['amber']}; margin-bottom:4px;">PERTURBATION</div>
        <img src="data:image/png;base64,{noise_b64}"
             style="border-radius:8px; border:2px solid {COLORS['amber']};">
        <div style="font-size:11px; margin-top:6px; color:{COLORS['muted']};">δ stretched to [0,1]</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px; color:{COLORS['red']}; margin-bottom:4px;">ADVERSARIAL</div>
        <img src="data:image/png;base64,{adv_b64}"
             style="border-radius:8px; border:2px solid {COLORS['red']};">
        <div style="font-size:11px; margin-top:6px;">{_esc(adv_label)}</div>
        <div style="font-size:10px; color:{COLORS['muted']};">{adv_prob:.1%}</div>
      </div>
    </div>

    <div style="display:flex; justify-content:center;">
      <img src="data:image/png;base64,{plot_b64}"
           style="border-radius:6px; border:1px solid {COLORS['border']};">
    </div>
    """
    return _shell(inner, max_width=900)


def render_vlm_final(
    *,
    model_name: str,
    target_label: str,
    prompt: str,
    target_text: str,
    baseline_response: str,
    final_response: str,
    original_np: np.ndarray,
    adv_np: np.ndarray,
    noise_np: np.ndarray,
    best_loss: float,
    linf: float,
    psnr: Optional[float],
) -> str:
    orig_b64 = _np_to_b64(original_np, (240, 240))
    adv_b64 = _np_to_b64(adv_np, (240, 240))
    noise_b64 = _np_to_b64(noise_np, (240, 240))
    psnr_label = f"{psnr:.1f}dB" if psnr is not None else "—"

    inner = f"""
    <h3 style="color:{COLORS['blue']}; margin:0 0 8px 0;">
      VLM PGD — {_esc(target_label)}
    </h3>
    <div style="font-size:11px; color:{COLORS['muted']}; margin-bottom:14px;">
      {_esc(model_name)} · best loss = {best_loss:.4f} · L∞ = {linf:.4f} · PSNR = {psnr_label}
    </div>

    <div style="display:flex; gap:12px; margin-bottom:14px; justify-content:center;">
      <div style="text-align:center;">
        <div style="font-size:10px; color:{COLORS['green']}; margin-bottom:4px;">CLEAN</div>
        <img src="data:image/png;base64,{orig_b64}"
             style="border-radius:8px; border:2px solid {COLORS['green']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px; color:{COLORS['amber']}; margin-bottom:4px;">PERTURBATION (stretched)</div>
        <img src="data:image/png;base64,{noise_b64}"
             style="border-radius:8px; border:2px solid {COLORS['amber']};">
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px; color:{COLORS['red']}; margin-bottom:4px;">ADVERSARIAL</div>
        <img src="data:image/png;base64,{adv_b64}"
             style="border-radius:8px; border:2px solid {COLORS['red']};">
      </div>
    </div>

    <div style="font-size:11px; background:{COLORS['panel']}; padding:8px 10px;
                border-radius:6px; margin-bottom:10px; border-left:3px solid {COLORS['blue']};">
      <b style="color:{COLORS['blue']};">Prompt:</b> {_esc(prompt)}
    </div>
    <div style="font-size:11px; background:{COLORS['panel']}; padding:8px 10px;
                border-radius:6px; margin-bottom:10px; border-left:3px solid {COLORS['amber']};">
      <b style="color:{COLORS['amber']};">Target:</b> {_esc(target_text[:200])}{'…' if len(target_text) > 200 else ''}
    </div>
    <div style="display:flex; gap:12px;">
      <div style="flex:1; background:{COLORS['panel']}; padding:10px; border-radius:8px;
                  border-left:3px solid {COLORS['green']};">
        <div style="font-size:10px; color:{COLORS['green']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:4px;">Clean response</div>
        <div style="font-size:12px; white-space:pre-wrap;">{_esc(baseline_response)}</div>
      </div>
      <div style="flex:1; background:{COLORS['panel']}; padding:10px; border-radius:8px;
                  border-left:3px solid {COLORS['red']};">
        <div style="font-size:10px; color:{COLORS['red']}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:4px;">Adversarial response</div>
        <div style="font-size:12px; white-space:pre-wrap;">{_esc(final_response)}</div>
      </div>
    </div>
    """
    return _shell(inner, max_width=1100)


def render_defense_compare(
    *,
    defense_name: str,
    description: str,
    badges: List[Tuple[str, str]],
    clean_label: str,
    clean_status: str,
    clean_color: str,
    adv_label: str,
    adv_status: str,
    adv_color: str,
    clean_image: Optional[Image.Image] = None,
    adv_image: Optional[Image.Image] = None,
    extras_html: str = "",
) -> str:
    badge_html = " ".join(
        f'<span style="background:{COLORS["panel"]}; padding:3px 10px; '
        f'border-radius:4px; color:{COLORS["muted"]}; font-size:11px;">'
        f'{_esc(k)}: <b style="color:{COLORS["amber"]};">{_esc(str(v))}</b></span>'
        for k, v in badges
    )

    def _img(im, color):
        if im is None:
            return ""
        b64 = _pil_to_b64(im, (220, 220))
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="border-radius:6px; border:1px solid {color}; margin-bottom:8px;">')

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
                  border-left:4px solid {clean_color}; text-align:center;">
        <div style="font-size:10px; color:{clean_color}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">✅ CLEAN — {_esc(clean_label)}</div>
        {_img(clean_image, clean_color)}
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    white-space:pre-wrap; min-height:60px;">{_esc(clean_status)}</div>
      </div>
      <div style="flex:1; background:{COLORS['panel']}; border-radius:8px; padding:12px;
                  border-left:4px solid {adv_color}; text-align:center;">
        <div style="font-size:10px; color:{adv_color}; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:6px;">⚠️ ADVERSARIAL — {_esc(adv_label)}</div>
        {_img(adv_image, adv_color)}
        <div style="font-size:12px; background:{COLORS['bg']}; padding:8px; border-radius:4px;
                    white-space:pre-wrap; min-height:60px;">{_esc(adv_status)}</div>
      </div>
    </div>
    {extras_html}
    """
    return _shell(inner, max_width=1100)


def render_variant_panel(
    *,
    variant_name: str,
    paper_citation: str,
    description: str,
    image_columns: List[dict],
    extras_html: str = "",
) -> str:
    parts = []
    for col in image_columns:
        b64 = _pil_to_b64(col["image"], (220, 220)) if "image" in col \
              else _np_to_b64(col["image_np"], (220, 220))
        parts.append(f"""
        <div style="text-align:center;">
          <div style="font-size:10px; color:{COLORS['muted']}; margin-bottom:4px;">
            {_esc(col.get('label', ''))}
          </div>
          <img src="data:image/png;base64,{b64}"
               style="border-radius:6px; border:1px solid {COLORS['amber']}; background:#fff;">
          <div style="font-size:10px; color:{COLORS['muted']}; margin-top:4px;">
            {_esc(col.get('caption', ''))}
          </div>
        </div>
        """)
    images_html = ('<div style="display:flex; gap:14px; justify-content:center; '
                   'flex-wrap:wrap; margin-bottom:14px;">' + "".join(parts) + "</div>")

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
    {extras_html}
    """
    return _shell(inner, max_width=1100)


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
    return f'<div style="display:flex; gap:8px; margin:10px 0;">' + "".join(items) + "</div>"
