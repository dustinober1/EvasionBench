from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_label_distribution(series, title="Label distribution"):
    ax = sns.countplot(x=series)
    ax.set_title(title)
    return ax


def plot_macro_f1_comparison(ranking: pd.DataFrame, output_path: str | Path) -> Path:
    chart = ranking[["model_family", "f1_macro"]].copy().sort_values(by="f1_macro", ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=chart, x="model_family", y="f1_macro", palette="Blues_d")
    plt.ylim(0.0, 1.0)
    plt.title("Phase 5 Macro-F1 by Model Family")
    plt.xlabel("Model Family")
    plt.ylabel("Macro F1")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_per_class_delta_heatmap(per_class_wide: pd.DataFrame, output_path: str | Path) -> Path:
    score_cols = [col for col in per_class_wide.columns if col != "label"]
    matrix = per_class_wide.set_index("label")[score_cols]
    best = matrix.max(axis=1)
    deltas = matrix.sub(best, axis=0)

    plt.figure(figsize=(8, max(3, len(deltas) * 0.7)))
    sns.heatmap(deltas, annot=True, fmt=".3f", cmap="RdBu", center=0.0, cbar=True)
    plt.title("Per-Class F1 Delta vs Best Family")
    plt.xlabel("Model Family")
    plt.ylabel("Label")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out
