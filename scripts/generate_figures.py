"""Rebuild figures only from reviewed, archived notebook values."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / "evidence/recorded_metrics.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.dpi": 180,
        }
    )
    colors = ["#2563a6", "#64748b", "#bc5a18", "#2563a6", "#64748b", "#bc5a18"]
    labels = [r["name"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    y = np.arange(len(rows))
    for ax, metric, title in zip(
        axes, ["accuracy", "f1"], ["Test accuracy", "Test fall F1"], strict=True
    ):
        values = [r[metric] * 100 for r in rows]
        ax.barh(y, values, color=colors, height=0.6)
        for i, value in enumerate(values):
            ax.text(value + 1.0, i, f"{value:.2f}%", va="center", fontsize=10)
        ax.set_xlim(0, 107)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Percent")
        ax.set_title(title, loc="left", pad=15)
        ax.grid(axis="x", alpha=0.17)
        ax.set_axisbelow(True)
        ax.set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Six notebook experiments · recorded test results",
        x=0.025,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.015,
        "Source: notebook cells 8, 10, 12, 21, 22, 24 | 2,572 windows | Rounded saved values; not reproduced runs",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.015, 0.06, 1, 0.93), w_pad=3)
    fig.savefig(OUT / "recorded_results.png", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.3))
    adl = np.array([5063, 687, 1891])
    falls = np.array([1811, 255, 681])
    ypos = np.arange(3)
    ax.barh(ypos, adl, color="#2563a6", label="ADL")
    ax.barh(ypos, falls, left=adl, color="#bc5a18", label="Fall")
    for i in ypos:
        ax.text(
            adl[i] / 2, i, f"{adl[i]:,}", ha="center", va="center", color="white", fontweight="bold"
        )
        ax.text(
            adl[i] + falls[i] / 2,
            i,
            f"{falls[i]:,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8 if falls[i] < 400 else 11,
        )
        ax.text(adl[i] + falls[i] + 100, i, f"{adl[i] + falls[i]:,}", va="center", color="#334155")
    ax.set_yticks(ypos, ["Train · 26 subjects", "Validation · 2 subjects", "Test · 8 subjects"])
    ax.invert_yaxis()
    ax.set_xlim(0, 7700)
    ax.set_xlabel("Recorded windows")
    ax.set_title("Fixed subject split · ADL and fall windows", loc="left", pad=20)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["left"].set_visible(False)
    fig.text(
        0.025,
        0.02,
        "Source: notebook cell 5 | 10,388 selected windows; 2,689 additional windows excluded by subject lists",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "subject_split.png", facecolor="white")
    plt.close(fig)

    history = json.loads((ROOT / "evidence/validation_history.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, family in enumerate([history[:3], history[3:]]):
        for col_index, field in enumerate(["accuracy", "f1"]):
            ax = axes[row_index, col_index]
            for rec, color in zip(family, ["#2563a6", "#64748b", "#bc5a18"], strict=True):
                vals = rec["validation_history"]
                ax.plot(
                    [v["epoch"] for v in vals],
                    [v[field] * 100 for v in vals],
                    color=color,
                    label=rec["name"],
                    linewidth=1.8,
                )
                selection = "f1" if rec["experiment"] == "lighthart_kd" else "accuracy"
                chosen = max(vals, key=lambda v: v[selection])
                ax.scatter(
                    [chosen["epoch"]],
                    [chosen[field] * 100],
                    s=38,
                    color=color,
                    edgecolor="white",
                    zorder=3,
                )
            ax.set_xlabel("Recorded epoch")
            ax.set_ylabel(
                "Validation " + ("accuracy (%)" if field == "accuracy" else "fall F1 (%)")
            )
            ax.set_ylim(20 if field == "f1" else 65, 100)
            ax.grid(alpha=0.18)
            ax.legend(fontsize=8, frameon=False, loc="lower right")
            ax.set_title(
                ("IT family" if row_index == 0 else "LightHART family")
                + " · "
                + ("accuracy" if field == "accuracy" else "fall F1"),
                loc="left",
            )
    fig.suptitle(
        "Validation histories from the original notebook",
        x=0.025,
        ha="left",
        fontweight="bold",
        fontsize=17,
    )
    fig.text(
        0.025,
        0.015,
        "Dots: checkpoint selected by the source criterion. Thresholds maximize validation accuracy each epoch. Curves use saved rounded values.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=2.4)
    fig.savefig(OUT / "validation_history.png", facecolor="white")
    plt.close(fig)
    print("Wrote three figures from archived notebook values; no new benchmark experiments")


if __name__ == "__main__":
    main()
