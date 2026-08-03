import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# PATHS 
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ALL RESULTS, manually compiled from training outputs
# accuracy results from training
ACCURACY_RESULTS = {
    "LLVIP": {
        "RGB Baseline":            {"mAP50": 0.885, "mAP5095": 0.506},
        "Thermal Baseline":        {"mAP50": 0.958, "mAP5095": 0.643},
        "Early Fusion":            {"mAP50": 0.894, "mAP5095": 0.520},
        "Intermediate Fusion":     {"mAP50": 0.896, "mAP5095": 0.522},
        "Late Fusion":             {"mAP50": 0.962, "mAP5095": 0.640},
    },
    "FLIR": {
        "RGB Baseline":            {"mAP50": 0.524, "mAP5095": 0.206},
        "Thermal Baseline":        {"mAP50": 0.657, "mAP5095": 0.295},
        "Early Fusion":            {"mAP50": 0.518, "mAP5095": 0.202},
        "Intermediate Fusion":     {"mAP50": 0.524, "mAP5095": 0.204},
        "Late Fusion":             {"mAP50": 0.690, "mAP5095": 0.299},
    }
}

# edge efficiency results from benchmarking
EFFICIENCY_RESULTS = {
    "RGB Baseline":        {"size_mb": 11.7,  "latency_ms": 25.0,  "params_m": 3.01, "flops_g": 4.1},
    "Thermal Baseline":    {"size_mb": 11.7,  "latency_ms": 26.0,  "params_m": 3.01, "flops_g": 4.1},
    "Early Fusion":        {"size_mb": 11.7,  "latency_ms": 27.7,  "params_m": 3.01, "flops_g": 4.1},
    "Intermediate Fusion": {"size_mb": 11.7,  "latency_ms": 27.7,  "params_m": 3.01, "flops_g": 4.1},
    "Late Fusion":         {"size_mb": 23.4,  "latency_ms": 50.8,  "params_m": 6.02, "flops_g": 8.2},
}

# colour scheme for consistent plotting
COLORS = {
    "RGB Baseline":        "#4e79a7",
    "Thermal Baseline":    "#f28e2b",
    "Early Fusion":        "#e15759",
    "Intermediate Fusion": "#76b7b2",
    "Late Fusion":         "#59a14f",
}

MODEL_ORDER = ["RGB Baseline", "Thermal Baseline", "Early Fusion",
               "Intermediate Fusion", "Late Fusion"]


# FIGURE 1: mAP50 Comparison Bar Chart
# Shows accuracy of all models side by side on both datasets

def plot_map50_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("mAP50 Comparison Across Fusion Strategies",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, dataset in zip(axes, ["LLVIP", "FLIR"]):
        models  = MODEL_ORDER
        values  = [ACCURACY_RESULTS[dataset][m]["mAP50"] for m in models]
        colors  = [COLORS[m] for m in models]

        bars = ax.bar(range(len(models)), values, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.6)

        # add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        # add horizontal line for thermal baseline (target to beat)
        thermal_val = ACCURACY_RESULTS[dataset]["Thermal Baseline"]["mAP50"]
        ax.axhline(y=thermal_val, color="#f28e2b", linestyle="--",
                   linewidth=1.5, alpha=0.7, label=f"Thermal baseline ({thermal_val:.3f})")

        ax.set_title(f"{dataset} Dataset", fontsize=13, fontweight="bold")
        ax.set_ylabel("mAP50", fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "fig1_map50_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# FIGURE 2: Accuracy vs Efficiency Scatter Plot
# Shows the trade-off between mAP50 and inference latency
# This is the key figure for edge deployment analysis

def plot_accuracy_efficiency_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Accuracy vs Latency Trade-off (Edge Deployment)",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, dataset in zip(axes, ["LLVIP", "FLIR"]):
        for model in MODEL_ORDER:
            acc     = ACCURACY_RESULTS[dataset][model]["mAP50"]
            latency = EFFICIENCY_RESULTS[model]["latency_ms"]
            color   = COLORS[model]

            ax.scatter(latency, acc, s=200, color=color,
                       zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(model,
                        xy=(latency, acc),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                        fontweight="bold")

        ax.set_title(f"{dataset} Dataset", fontsize=13, fontweight="bold")
        ax.set_xlabel("Inference Latency (ms) — CPU", fontsize=11)
        ax.set_ylabel("mAP50", fontsize=11)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # add annotation explaining ideal quadrant
        ax.text(0.02, 0.98, "← Faster\n↑ More accurate\n(ideal: top-left)",
                transform=ax.transAxes, fontsize=8,
                va="top", color="gray", style="italic")

    plt.tight_layout()
    path = RESULTS_DIR / "fig2_accuracy_efficiency_tradeoff.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# FIGURE 3 Edge Efficiency Metrics Bar Chart
# Shows size, latency, params and FLOPs for all models

def plot_efficiency_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Edge Efficiency Metrics by Fusion Strategy",
                 fontsize=15, fontweight="bold")

    metrics = [
        ("size_mb",    "Model Size (MB)",          axes[0][0]),
        ("latency_ms", "Inference Latency (ms)",   axes[0][1]),
        ("params_m",   "Parameters (Millions)",    axes[1][0]),
        ("flops_g",    "FLOPs (Giga)",             axes[1][1]),
    ]

    for key, title, ax in metrics:
        models = MODEL_ORDER
        values = [EFFICIENCY_RESULTS[m][key] for m in models]
        colors = [COLORS[m] for m in models]

        bars = ax.bar(range(len(models)), values, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.6)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.01,
                    f"{val}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # add legend
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODEL_ORDER]
    fig.legend(handles=patches, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=9)

    plt.tight_layout()
    path = RESULTS_DIR / "fig3_efficiency_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# FIGURE 4: mAP50 vs mAP50-95 Grouped Bar Chart
# Shows both metrics side by side for deeper accuracy analysis

def plot_dual_metric_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("mAP50 vs mAP50-95 Comparison",
                 fontsize=15, fontweight="bold", y=1.02)

    x      = np.arange(len(MODEL_ORDER))
    width  = 0.35

    for ax, dataset in zip(axes, ["LLVIP", "FLIR"]):
        map50_vals   = [ACCURACY_RESULTS[dataset][m]["mAP50"]   for m in MODEL_ORDER]
        map5095_vals = [ACCURACY_RESULTS[dataset][m]["mAP5095"] for m in MODEL_ORDER]

        bars1 = ax.bar(x - width/2, map50_vals,   width,
                       label="mAP50",    color="#4e79a7", alpha=0.85)
        bars2 = ax.bar(x + width/2, map5095_vals, width,
                       label="mAP50-95", color="#f28e2b", alpha=0.85)

        for bar, val in zip(bars1, map50_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        for bar, val in zip(bars2, map5095_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{dataset} Dataset", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = RESULTS_DIR / "fig4_dual_metric_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# PRINT SUMMARY TABLES
# Clean formatted tables for copying into report

def print_summary_tables():
    print("\n" + "="*70)
    print("  TABLE 1 — ACCURACY RESULTS")
    print("="*70)
    print(f"  {'Model':<25} {'LLVIP mAP50':>12} {'LLVIP mAP5095':>14} "
          f"{'FLIR mAP50':>11} {'FLIR mAP5095':>13}")
    print(f"  {'-'*70}")

    for model in MODEL_ORDER:
        ll50   = ACCURACY_RESULTS["LLVIP"][model]["mAP50"]
        ll5095 = ACCURACY_RESULTS["LLVIP"][model]["mAP5095"]
        fl50   = ACCURACY_RESULTS["FLIR"][model]["mAP50"]
        fl5095 = ACCURACY_RESULTS["FLIR"][model]["mAP5095"]
        print(f"  {model:<25} {ll50:>12.3f} {ll5095:>14.3f} "
              f"{fl50:>11.3f} {fl5095:>13.3f}")

    print("\n" + "="*70)
    print("  TABLE 2 — EDGE EFFICIENCY RESULTS")
    print("="*70)
    print(f"  {'Model':<25} {'Size (MB)':>10} {'Latency (ms)':>13} "
          f"{'Params (M)':>11} {'FLOPs (G)':>10}")
    print(f"  {'-'*70}")

    for model in MODEL_ORDER:
        e = EFFICIENCY_RESULTS[model]
        print(f"  {model:<25} {e['size_mb']:>10.1f} {e['latency_ms']:>13.1f} "
              f"{e['params_m']:>11.2f} {e['flops_g']:>10.1f}")

    print("\n" + "="*70)
    print("  TABLE 3 — IMPROVEMENT OVER RGB BASELINE (%)")
    print("="*70)
    print(f"  {'Model':<25} {'LLVIP gain':>12} {'FLIR gain':>12}")
    print(f"  {'-'*50}")

    rgb_ll = ACCURACY_RESULTS["LLVIP"]["RGB Baseline"]["mAP50"]
    rgb_fl = ACCURACY_RESULTS["FLIR"]["RGB Baseline"]["mAP50"]

    for model in MODEL_ORDER:
        ll50 = ACCURACY_RESULTS["LLVIP"][model]["mAP50"]
        fl50 = ACCURACY_RESULTS["FLIR"][model]["mAP50"]
        ll_gain = ((ll50 - rgb_ll) / rgb_ll) * 100
        fl_gain = ((fl50 - rgb_fl) / rgb_fl) * 100
        print(f"  {model:<25} {ll_gain:>+11.1f}% {fl_gain:>+11.1f}%")


# MAIN

if __name__ == "__main__":
    print("\n Generating Results Analysis")
    print("   Creating figures and tables for report\n")

    print("  Generating figures:")
    plot_map50_comparison()
    plot_accuracy_efficiency_tradeoff()
    plot_efficiency_metrics()
    plot_dual_metric_comparison()

    print_summary_tables()

    print(f"\n All figures saved to: {RESULTS_DIR}")
    print("   fig1_map50_comparison.png")
    print("   fig2_accuracy_efficiency_tradeoff.png")
    print("   fig3_efficiency_metrics.png")
    print("   fig4_dual_metric_comparison.png")
    print("\n Analysis Complete — ready for report writing!")