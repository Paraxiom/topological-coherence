#!/usr/bin/env python3
"""Generate v2 benchmark charts for the Toroidal Logit Bias paper.

Produces 3 PNGs from full_benchmark_results_v2.json:
  1. truthfulqa_v2_bar.png      — Grouped bar: 4 models × baseline vs toroidal (T&I %)
  2. truthfulqa_v2_scaling.png  — Line: model size vs T&I %, two lines
  3. truthfulqa_v2_breakdown.png — Stacked bar: category breakdown per model×method
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
DATA = HERE / "full_benchmark_results_v2.json"

# Paraxiom palette
GREEN = "#2d5a27"
GOLD = "#a08a62"
GRAY = "#666666"
LIGHT_GREEN = "#6a9a6a"
BG = "#fafafa"

# Model display names and sizes (for scaling chart)
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen 0.5B", 0.5),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen 1.5B", 1.5),
    ("Qwen/Qwen2.5-7B-Instruct",   "Qwen 7B",   7),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B", 7),
]


def load_data():
    with open(DATA) as f:
        return json.load(f)["results"]


# ── Chart 1: Grouped bar chart ──────────────────────────────────────────────

def chart_bar(data):
    labels = [m[1] for m in MODELS]
    baseline_vals = [data[m[0]]["baseline"]["truthful_and_informative_pct"] for m in MODELS]
    toroidal_vals = [data[m[0]]["toroidal"]["truthful_and_informative_pct"] for m in MODELS]
    deltas = [t - b for b, t in zip(baseline_vals, toroidal_vals)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars_b = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color=GRAY, edgecolor="white", linewidth=0.5)
    bars_t = ax.bar(x + width / 2, toroidal_vals, width, label="Toroidal", color=GREEN, edgecolor="white", linewidth=0.5)

    # Delta labels
    for i, (bar, delta) in enumerate(zip(bars_t, deltas)):
        ax.annotate(
            f"+{delta:.1f}pp",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=GREEN,
        )

    ax.set_ylabel("Truthful & Informative (%)", fontsize=12)
    ax.set_title("TruthfulQA v2 — Baseline vs Toroidal (817 samples, LLM-judged)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(toroidal_vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = HERE / "truthfulqa_v2_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


# ── Chart 2: Scaling line chart (Qwen only: 0.5B, 1.5B, 7B) ────────────────

def chart_scaling(data):
    qwen_models = MODELS[:3]  # Only Qwen for scaling
    sizes = [m[2] for m in qwen_models]
    labels = [m[1] for m in qwen_models]
    baseline_vals = [data[m[0]]["baseline"]["truthful_and_informative_pct"] for m in qwen_models]
    toroidal_vals = [data[m[0]]["toroidal"]["truthful_and_informative_pct"] for m in qwen_models]
    deltas = [t - b for b, t in zip(baseline_vals, toroidal_vals)]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)

    ax1.plot(sizes, baseline_vals, "o-", color=GRAY, linewidth=2.5, markersize=10, label="Baseline", zorder=3)
    ax1.plot(sizes, toroidal_vals, "s-", color=GREEN, linewidth=2.5, markersize=10, label="Toroidal", zorder=3)

    # Shade the delta region
    ax1.fill_between(sizes, baseline_vals, toroidal_vals, alpha=0.15, color=GREEN)

    # Annotate deltas
    for s, b, t, d in zip(sizes, baseline_vals, toroidal_vals, deltas):
        ax1.annotate(
            f"+{d:.1f}pp",
            xy=(s, (b + t) / 2),
            xytext=(15, 0), textcoords="offset points",
            ha="left", va="center", fontsize=10, fontweight="bold", color=GREEN,
        )

    ax1.set_xlabel("Model Size (Billion Parameters)", fontsize=12)
    ax1.set_ylabel("Truthful & Informative (%)", fontsize=12)
    ax1.set_title("Toroidal Improvement Scales with Model Size", fontsize=14, fontweight="bold")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    out = HERE / "truthfulqa_v2_scaling.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


# ── Chart 3: Stacked bar breakdown ──────────────────────────────────────────

def chart_breakdown(data):
    # Only baseline + toroidal (public)
    methods = ["baseline", "toroidal"]
    method_labels = ["Baseline", "Toroidal"]

    categories = ["truthful_and_informative_pct", "truthful_only_pct", "informative_only_pct", "neither_pct"]
    cat_labels = ["Truthful & Informative", "Truthful Only", "Informative Only", "Neither"]
    cat_colors = [GREEN, LIGHT_GREEN, GOLD, GRAY]

    group_labels = []
    cat_data = {c: [] for c in categories}

    for model_key, model_name, _ in MODELS:
        for method, mlabel in zip(methods, method_labels):
            group_labels.append(f"{model_name}\n{mlabel}")
            for cat in categories:
                cat_data[cat].append(data[model_key][method][cat])

    x = np.arange(len(group_labels))
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bottom = np.zeros(len(group_labels))
    for cat, label, color in zip(categories, cat_labels, cat_colors):
        vals = np.array(cat_data[cat])
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5, width=0.7)
        bottom += vals

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("TruthfulQA v2 — Response Category Breakdown (817 samples)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9, rotation=0, ha="center")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add vertical separators between model groups
    for i in range(1, len(MODELS)):
        ax.axvline(x=i * 2 - 0.5, color="#cccccc", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out = HERE / "truthfulqa_v2_breakdown.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    print("Loading benchmark data...")
    data = load_data()
    print("Generating charts:")
    chart_bar(data)
    chart_scaling(data)
    chart_breakdown(data)
    print("Done.")
