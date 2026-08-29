"""検証指標の推移プロット(best 更新時に 1 枚の PNG を上書き出力)。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz 既定パレット(light)
_COLORS = {"300wlp_val": "#2a78d6", "300w_vc": "#eb6834",
           "wflw_test": "#1baf7a", "cofw_test": "#eda100"}
_MEAN = "#0b0b0b"
_SURFACE = "#fcfcfb"
_TEXT2 = "#52514e"


def _style(ax, title, ylabel):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, color=_MEAN)
    ax.set_xlabel("epoch", color=_TEXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=_TEXT2, fontsize=9)
    ax.tick_params(colors=_TEXT2, labelsize=8)


def plot_val_metrics(history: list[dict], out_path: Path) -> None:
    """history: [{"epoch", "val": {set: {head_nme, vis_acc, pose_err_deg?}},
    "val_mean_nme"}, ...] を 3 面プロットして out_path に上書き保存する。"""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    sets = list(history[-1]["val"].keys())

    fig, axes2d = plt.subplots(2, 2, figsize=(10.5, 7.2), facecolor=_SURFACE)
    axes = axes2d.ravel()

    _style(axes[0], "head-NME (crop-normalized)", "NME")
    for s in sets:
        axes[0].plot(epochs, [h["val"][s]["head_nme"] for h in history],
                     color=_COLORS.get(s, _TEXT2), linewidth=2, label=s)
    axes[0].plot(epochs, [h["val_mean_nme"] for h in history],
                 color=_MEAN, linewidth=2, linestyle="--", label="mean")
    best = min(history, key=lambda h: h["val_mean_nme"])
    axes[0].scatter([best["epoch"]], [best["val_mean_nme"]], color=_MEAN,
                    zorder=5, s=24)
    axes[0].legend(frameon=False, fontsize=8)

    _style(axes[1], "visibility accuracy", "accuracy")
    for s in sets:
        axes[1].plot(epochs, [h["val"][s]["vis_acc"] for h in history],
                     color=_COLORS.get(s, _TEXT2), linewidth=2, label=s)
    axes[1].legend(frameon=False, fontsize=8)

    _style(axes[2], "pose geodesic error (300wlp_val)", "deg")
    pe = [(h["epoch"], h["val"]["300wlp_val"].get("pose_err_deg"))
          for h in history if "300wlp_val" in h["val"]]
    pe = [(e, v) for e, v in pe if v is not None]
    if pe:
        axes[2].plot([e for e, _ in pe], [v for _, v in pe],
                     color=_COLORS["300wlp_val"], linewidth=2)

    _style(axes[3], "processing time per epoch", "sec")
    for key, color, label in [("epoch_train_sec", "#4a3aa7", "train"),
                              ("eval_sec", "#e87ba4", "eval")]:
        tt = [(h["epoch"], h.get(key)) for h in history if h.get(key) is not None]
        if tt:
            axes[3].plot([e for e, _ in tt], [v for _, v in tt],
                         color=color, linewidth=2, label=label)
    if axes[3].lines:
        axes[3].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor=_SURFACE)
    plt.close(fig)
