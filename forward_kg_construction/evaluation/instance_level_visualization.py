"""
Instance-Level Metrics Visualization

Visualizations for instance-level agreement metrics with Jaccard weighting.
Generates academic-quality plots for:
- Agreement distribution (Agree/Partial/Disagree)
- Jaccard score distribution
- Summary metrics dashboard
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Pastel color palette for academic papers
PASTEL_COLORS = {
    "blue": "#A8D5E5",
    "green": "#B5E5A8",
    "orange": "#F5D5A8",
    "pink": "#E5A8C8",
    "purple": "#C8A8E5",
    "teal": "#A8E5D5",
    "red": "#E5A8A8",
    "yellow": "#E5E5A8",
}

# Agreement category colors
AGREEMENT_COLORS = {
    "agree": "#B5E5A8",      # Green - full agreement
    "partial": "#F5D5A8",    # Orange - partial agreement
    "disagree": "#E5A8A8",   # Red - disagreement
}


def plot_instance_level_metrics(
    metrics: Dict[str, Any],
    output_dir: str = "./evaluation_output",
    dpi: int = 300,
    prefix: str = "instance_level",
) -> List[str]:
    """
    Generate academic-quality plots for instance-level agreement metrics.

    Args:
        metrics: Dictionary from calculate_jaccard_agreement_score containing:
            - n_total, n_agree, n_partial, n_disagree
            - agree_rate, partial_rate, disagree_rate
            - avg_jaccard_score, avg_partial_jaccard
            - jaccard_scores (list)
        output_dir: Directory to save plots
        dpi: Resolution for PNG output (default: 300 for publication quality)
        prefix: Prefix for output filenames

    Returns:
        List of paths to generated plot files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_files = []

    # Set matplotlib style for academic papers
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # =========================================================================
    # Plot 1: Agreement Distribution Pie Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    labels = ["Agree", "Partial", "Disagree"]
    values = [
        metrics.get("n_agree", 0),
        metrics.get("n_partial", 0),
        metrics.get("n_disagree", 0),
    ]
    colors = [AGREEMENT_COLORS["agree"], AGREEMENT_COLORS["partial"], AGREEMENT_COLORS["disagree"]]

    if sum(values) > 0:
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(values))})",
            colors=colors,
            startangle=90,
            explode=(0.02, 0.02, 0.02),
            wedgeprops={"edgecolor": "gray", "linewidth": 0.8},
            textprops={"fontsize": 11},
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("medium")
        ax.set_title(f"Agreement Distribution (n={metrics.get('n_total', 0)})")
    else:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Agreement Distribution")

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_agreement_pie_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 2: Agreement Distribution Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="gray", linewidth=0.8, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Count")
    ax.set_title(f"Agreement Category Distribution (n={metrics.get('n_total', 0)})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.15 if values and max(values) > 0 else 1)

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_agreement_bar_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 3: Jaccard Score Distribution Histogram
    # =========================================================================
    jaccard_scores = metrics.get("jaccard_scores", [])

    if jaccard_scores:
        fig, ax = plt.subplots(figsize=(9, 5))

        # Create histogram with bins from 0 to 1
        bins = np.linspace(0, 1, 11)  # 10 bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0
        n, bins_edges, patches = ax.hist(
            jaccard_scores, bins=bins, color=PASTEL_COLORS["blue"],
            edgecolor="gray", linewidth=0.8, alpha=0.8
        )

        # Color the bars based on Jaccard score ranges
        for i, patch in enumerate(patches):
            if bins_edges[i] >= 0.8:
                patch.set_facecolor(AGREEMENT_COLORS["agree"])
            elif bins_edges[i] >= 0.3:
                patch.set_facecolor(AGREEMENT_COLORS["partial"])
            else:
                patch.set_facecolor(AGREEMENT_COLORS["disagree"])

        # Add mean line
        mean_jaccard = metrics.get("avg_jaccard_score", np.mean(jaccard_scores))
        ax.axvline(mean_jaccard, color="#E74C3C", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_jaccard:.3f}")

        ax.set_xlabel("Jaccard Similarity Score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Jaccard Score Distribution (n={len(jaccard_scores)})")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plot_path = output_path / f"{prefix}_jaccard_histogram_{timestamp}.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        generated_files.append(str(plot_path))
        logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 4: Summary Metrics Dashboard
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Agreement Rates
    ax1 = axes[0]
    rate_labels = ["Agree", "Partial", "Disagree"]
    rate_values = [
        metrics.get("agree_rate", 0) * 100,
        metrics.get("partial_rate", 0) * 100,
        metrics.get("disagree_rate", 0) * 100,
    ]
    rate_colors = [AGREEMENT_COLORS["agree"], AGREEMENT_COLORS["partial"],
                   AGREEMENT_COLORS["disagree"]]

    bars1 = ax1.bar(rate_labels, rate_values, color=rate_colors,
                    edgecolor="gray", linewidth=0.8)
    for bar, val in zip(bars1, rate_values):
        height = bar.get_height()
        ax1.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Agreement Rates")
    ax1.set_ylim(0, 100)

    # Panel 2: Jaccard Scores
    ax2 = axes[1]
    jaccard_labels = ["Avg Jaccard", "Avg Partial\nJaccard"]
    jaccard_values = [
        metrics.get("avg_jaccard_score", 0),
        metrics.get("avg_partial_jaccard", 0),
    ]
    jaccard_colors = [PASTEL_COLORS["blue"], PASTEL_COLORS["orange"]]

    bars2 = ax2.bar(jaccard_labels, jaccard_values, color=jaccard_colors,
                    edgecolor="gray", linewidth=0.8)
    for bar, val in zip(bars2, jaccard_values):
        height = bar.get_height()
        ax2.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Jaccard Score")
    ax2.set_title("Jaccard Similarity Metrics")
    ax2.set_ylim(0, 1.15)

    # Panel 3: Instance Counts
    ax3 = axes[2]
    count_labels = ["Total", "Agree", "Partial", "Disagree"]
    count_values = [
        metrics.get("n_total", 0),
        metrics.get("n_agree", 0),
        metrics.get("n_partial", 0),
        metrics.get("n_disagree", 0),
    ]
    count_colors = [PASTEL_COLORS["purple"], AGREEMENT_COLORS["agree"],
                    AGREEMENT_COLORS["partial"], AGREEMENT_COLORS["disagree"]]

    bars3 = ax3.bar(count_labels, count_values, color=count_colors,
                    edgecolor="gray", linewidth=0.8)
    for bar, val in zip(bars3, count_values):
        height = bar.get_height()
        ax3.annotate(f"{val}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Count")
    ax3.set_title("Instance Counts")
    max_count = max(count_values) if count_values and max(count_values) > 0 else 1
    ax3.set_ylim(0, max_count * 1.15)

    plt.suptitle("Instance-Level Metrics Summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = output_path / f"{prefix}_summary_dashboard_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 5: Jaccard Score Box Plot by Agreement Category
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    n_agree = metrics.get("n_agree", 0)
    n_partial = metrics.get("n_partial", 0)
    n_disagree = metrics.get("n_disagree", 0)

    box_data = []
    box_labels = []
    box_colors = []

    if n_agree > 0:
        box_data.append([1.0] * n_agree)
        box_labels.append(f"Agree\n(n={n_agree})")
        box_colors.append(AGREEMENT_COLORS["agree"])

    if n_partial > 0:
        # Extract partial scores (0 < score < 1)
        partial_scores = [s for s in jaccard_scores if 0 < s < 1]
        if partial_scores:
            box_data.append(partial_scores)
        else:
            avg_partial = metrics.get("avg_partial_jaccard", 0.5)
            box_data.append([avg_partial] * n_partial)
        box_labels.append(f"Partial\n(n={n_partial})")
        box_colors.append(AGREEMENT_COLORS["partial"])

    if n_disagree > 0:
        box_data.append([0.0] * n_disagree)
        box_labels.append(f"Disagree\n(n={n_disagree})")
        box_colors.append(AGREEMENT_COLORS["disagree"])

    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("gray")

        ax.set_ylabel("Jaccard Score")
        ax.set_title("Jaccard Score Distribution by Agreement Category")
        ax.set_ylim(-0.05, 1.1)
    else:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Jaccard Score Distribution by Agreement Category")

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_jaccard_boxplot_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    logger.info(f"Generated {len(generated_files)} instance-level plots in {output_dir}")
    return generated_files


def plot_label_level_metrics(
    label_metrics: Dict[str, Any],
    relationship_types: List[str],
    output_dir: str = "./evaluation_output",
    dpi: int = 300,
    prefix: str = "label_level",
) -> List[str]:
    """
    Generate academic-quality plots for label-level evaluation metrics.

    Args:
        label_metrics: Dictionary from evaluate_complete containing:
            - per_type: Dict with tp, fp, fn, tn, precision, recall, f1_score, support
            - micro: Dict with precision, recall, f1_score
            - macro: Dict with precision, recall, f1_score
        relationship_types: List of relationship type names
        output_dir: Directory to save plots
        dpi: Resolution for PNG output (default: 300 for publication quality)
        prefix: Prefix for output filenames

    Returns:
        List of paths to generated plot files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_files = []

    # Set matplotlib style for academic papers
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    per_type = label_metrics.get("per_type", {})
    micro = label_metrics.get("micro", {})
    macro = label_metrics.get("macro", {})

    # =========================================================================
    # Plot 1: Per-Type Metrics Grouped Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(relationship_types))
    width = 0.25

    precisions = [per_type.get(t, {}).get("precision", 0) for t in relationship_types]
    recalls = [per_type.get(t, {}).get("recall", 0) for t in relationship_types]
    f1_scores = [per_type.get(t, {}).get("f1_score", 0) for t in relationship_types]

    bars1 = ax.bar(x - width, precisions, width, label="Precision",
                   color=PASTEL_COLORS["blue"], edgecolor="gray", linewidth=0.8)
    bars2 = ax.bar(x, recalls, width, label="Recall",
                   color=PASTEL_COLORS["green"], edgecolor="gray", linewidth=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label="F1-Score",
                   color=PASTEL_COLORS["orange"], edgecolor="gray", linewidth=0.8)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.0%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Per-Type Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(relationship_types)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_per_type_metrics_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 2: Micro vs Macro Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics_names = ["Precision", "Recall", "F1-Score"]
    micro_values = [
        micro.get("precision", 0),
        micro.get("recall", 0),
        micro.get("f1_score", 0),
    ]
    macro_values = [
        macro.get("precision", 0),
        macro.get("recall", 0),
        macro.get("f1_score", 0),
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, micro_values, width, label="Micro-Avg",
                   color=PASTEL_COLORS["blue"], edgecolor="gray", linewidth=0.8)
    bars2 = ax.bar(x + width/2, macro_values, width, label="Macro-Avg",
                   color=PASTEL_COLORS["purple"], edgecolor="gray", linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1%}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Micro vs Macro Averaged Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_micro_macro_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 3: Radar Chart for Per-Type F1 Scores
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = relationship_types
    N = len(categories)
    f1_vals = [per_type.get(t, {}).get("f1_score", 0) for t in categories]
    f1_vals += f1_vals[:1]  # Close the polygon

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.plot(angles, f1_vals, "o-", linewidth=2, color="#5DADE2", markersize=8)
    ax.fill(angles, f1_vals, alpha=0.25, color="#A8D5E5")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("F1-Score by Relationship Type", y=1.08)

    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_radar_chart_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 4: Confusion Matrix Heatmap (Per-Type TP/FP/FN/TN)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    from matplotlib.colors import LinearSegmentedColormap
    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel", ["#FFFFFF", "#A8D5E5", "#5DADE2"]
    )

    for i, rel_type in enumerate(relationship_types[:4]):  # Max 4 types
        ax = axes[i]
        stats = per_type.get(rel_type, {})
        tp = stats.get("tp", 0)
        fp = stats.get("fp", 0)
        fn = stats.get("fn", 0)
        tn = stats.get("tn", 0)

        conf_matrix = np.array([[tp, fp], [fn, tn]])
        labels = [["TP", "FP"], ["FN", "TN"]]

        im = ax.imshow(conf_matrix, cmap=pastel_cmap, aspect="auto")

        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, f"{labels[ii][jj]}\n{conf_matrix[ii, jj]}",
                        ha="center", va="center", fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Positive", "Negative"])
        ax.set_yticklabels(["Positive", "Negative"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{rel_type}")

    plt.suptitle("Confusion Matrices by Relationship Type", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = output_path / f"{prefix}_confusion_matrices_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 5: Support Distribution (Stacked Bar)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    tp_vals = [per_type.get(t, {}).get("tp", 0) for t in relationship_types]
    fp_vals = [per_type.get(t, {}).get("fp", 0) for t in relationship_types]
    fn_vals = [per_type.get(t, {}).get("fn", 0) for t in relationship_types]

    x = np.arange(len(relationship_types))
    width = 0.6

    ax.bar(x, tp_vals, width, label="True Positives",
           color=AGREEMENT_COLORS["agree"], edgecolor="gray", linewidth=0.8)
    ax.bar(x, fp_vals, width, bottom=tp_vals, label="False Positives",
           color=AGREEMENT_COLORS["partial"], edgecolor="gray", linewidth=0.8)
    ax.bar(x, fn_vals, width, bottom=np.array(tp_vals) + np.array(fp_vals),
           label="False Negatives", color=AGREEMENT_COLORS["disagree"],
           edgecolor="gray", linewidth=0.8)

    ax.set_ylabel("Count")
    ax.set_title("TP/FP/FN Distribution by Relationship Type")
    ax.set_xticks(x)
    ax.set_xticklabels(relationship_types)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_tp_fp_fn_distribution_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    # =========================================================================
    # Plot 6: Precision-Recall Trade-off Scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, rel_type in enumerate(relationship_types):
        p = per_type.get(rel_type, {}).get("precision", 0)
        r = per_type.get(rel_type, {}).get("recall", 0)
        ax.scatter(r, p, s=150, label=rel_type, zorder=5)
        ax.annotate(rel_type, (r, p), xytext=(5, 5), textcoords="offset points",
                    fontsize=9)

    # Add micro and macro points
    ax.scatter(micro.get("recall", 0), micro.get("precision", 0), s=200,
               marker="^", color="#E74C3C", label="Micro-Avg", zorder=6)
    ax.scatter(macro.get("recall", 0), macro.get("precision", 0), s=200,
               marker="s", color="#9B59B6", label="Macro-Avg", zorder=6)

    # Add F1 iso-lines
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        r_vals = np.linspace(0.01, 1, 100)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_vals = f1 * r_vals / (2 * r_vals - f1)
            p_vals = np.where(np.isfinite(p_vals), p_vals, np.nan)
        valid = (p_vals >= 0) & (p_vals <= 1) & np.isfinite(p_vals)
        ax.plot(r_vals[valid], p_vals[valid], "--", color="gray", alpha=0.5,
                linewidth=0.8)
        # Label the iso-line
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[np.argmin(np.abs(r_vals[valid] - 0.9))]
            ax.annotate(f"F1={f1}", (r_vals[idx], p_vals[idx]), fontsize=8,
                        color="gray")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Trade-off by Type")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / f"{prefix}_precision_recall_scatter_{timestamp}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(str(plot_path))
    logger.info(f"Saved: {plot_path}")

    logger.info(f"Generated {len(generated_files)} label-level plots in {output_dir}")
    return generated_files


def plot_complete_evaluation(
    results: Dict[str, Any],
    relationship_types: List[str],
    output_dir: str = "./evaluation_output",
    dpi: int = 300,
) -> Dict[str, List[str]]:
    """
    Generate all visualizations for complete two-level evaluation.

    Args:
        results: Dictionary from evaluate_complete containing:
            - instance_level: Instance-level metrics
            - label_level: Label-level metrics
        relationship_types: List of relationship type names
        output_dir: Directory to save plots
        dpi: Resolution for PNG output

    Returns:
        Dictionary with 'instance_level' and 'label_level' file lists
    """
    instance_metrics = results.get("instance_level", {})
    label_metrics = results.get("label_level", {})

    # Generate instance-level plots
    instance_files = plot_instance_level_metrics(
        instance_metrics,
        output_dir=output_dir,
        dpi=dpi,
        prefix="instance_level",
    )

    # Generate label-level plots
    label_files = plot_label_level_metrics(
        label_metrics,
        relationship_types=relationship_types,
        output_dir=output_dir,
        dpi=dpi,
        prefix="label_level",
    )

    logger.info(f"Generated {len(instance_files) + len(label_files)} total plots")

    return {
        "instance_level": instance_files,
        "label_level": label_files,
    }
