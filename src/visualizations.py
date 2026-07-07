"""visualizations.py - Computer vision visualizations for steel defect detection."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#fafafa",
        "text.color": "#fafafa",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


def denormalize(img: np.ndarray, mean=None, std=None) -> np.ndarray:
    """Reverse ImageNet normalization for display."""
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    img = img.copy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return np.clip(img, 0, 1)


def plot_sample_grid(images: np.ndarray, labels: np.ndarray,
                     class_names: list, n_per_class: int = 3) -> plt.Figure:
    """Display sample images in a grid, one row per class."""
    _style()
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(3 * n_per_class, 3 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for c in range(n_classes):
        class_mask = labels == c
        class_images = images[class_mask]
        indices = np.random.choice(len(class_images), min(n_per_class, len(class_images)), replace=False)
        for j, idx in enumerate(indices):
            ax = axes[c, j]
            img = denormalize(class_images[idx].transpose(1, 2, 0))
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(class_names[c], fontsize=10, fontweight="bold", color="#fafafa")
    fig.suptitle("NEU-DET Sample Images", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_class_distribution(class_counts: np.ndarray, class_names: list) -> plt.Figure:
    """Bar chart of class distribution."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#22d3ee", "#a78bfa", "#f97316", "#f43f5e", "#22c55e", "#fbbf24"]
    bars = ax.bar(class_names, class_counts, color=colors[:len(class_names)],
                  width=0.6, edgecolor="#333")
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            total = cm[i].sum()
            pct = val / total * 100 if total > 0 else 0
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > cm.max() / 2 else "#333")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_training_curves(history: Dict) -> plt.Figure:
    """Loss and accuracy curves for training and validation."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "o-", color="#22d3ee", linewidth=1.5, label="Train")
    ax1.plot(epochs, history["val_loss"], "o-", color="#f43f5e", linewidth=1.5, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--")

    ax2.plot(epochs, history["train_acc"], "o-", color="#22d3ee", linewidth=1.5, label="Train")
    ax2.plot(epochs, history["val_acc"], "o-", color="#f43f5e", linewidth=1.5, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--")

    fig.tight_layout()
    return fig


def plot_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray,
                         prediction: int, confidence: float,
                         true_label: Optional[int] = None,
                         class_names: list = None) -> plt.Figure:
    """Overlay Grad-CAM heatmap on original image."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Original image
    img_display = denormalize(image.transpose(1, 2, 0))
    axes[0].imshow(img_display)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    pred_name = class_names[prediction] if class_names else str(prediction)
    title = f"Overlay\nPredicted: {pred_name} ({confidence:.1%})"
    if true_label is not None:
        true_name = class_names[true_label] if class_names else str(true_label)
        correct = "✓" if prediction == true_label else "✗"
        title += f"\nTrue: {true_name} {correct}"
    axes[2].set_title(title, fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.tight_layout()
    return fig


def plot_per_class_accuracy(per_class_acc: Dict) -> plt.Figure:
    """Bar chart of per-class accuracy."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(per_class_acc.keys())
    accs = list(per_class_acc.values())
    colors = ["#22c55e" if a > 0.8 else "#fbbf24" if a > 0.5 else "#f43f5e" for a in accs]

    bars = ax.bar(names, accs, color=colors, width=0.6, edgecolor="#333")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.15])
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    return fig


def plot_model_benchmark(benchmarks: Dict[str, Dict]) -> plt.Figure:
    """Compare model size, latency, and accuracy."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    names = list(benchmarks.keys())
    sizes = [benchmarks[n]["model_size_ms"] if "model_size_ms" in benchmarks[n] else benchmarks[n]["model_size_mb"] for n in names]
    latencies = [benchmarks[n]["avg_inference_ms"] for n in names]
    params = [benchmarks[n]["param_count"] / 1e6 for n in names]

    colors = ["#22d3ee", "#a78bfa", "#f97316"]

    # Model size
    bars = axes[0].barh(names, sizes, color=colors[:len(names)], height=0.5)
    axes[0].set_xlabel("Size (MB)")
    axes[0].set_title("Model Size", fontsize=11, fontweight="bold")
    for bar, s in zip(bars, sizes):
        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{s:.1f} MB", va="center", fontsize=9)

    # Latency
    bars = axes[1].barh(names, latencies, color=colors[:len(names)], height=0.5)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("Inference Latency", fontsize=11, fontweight="bold")
    for bar, l in zip(bars, latencies):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{l:.1f} ms", va="center", fontsize=9)

    # Parameters
    bars = axes[2].barh(names, params, color=colors[:len(names)], height=0.5)
    axes[2].set_xlabel("Parameters (M)")
    axes[2].set_title("Parameter Count", fontsize=11, fontweight="bold")
    for bar, p in zip(bars, params):
        axes[2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{p:.2f}M", va="center", fontsize=9)

    for ax in axes:
        ax.grid(axis="x", linestyle="--")
    fig.tight_layout()
    return fig


def plot_feature_maps(feature_maps: np.ndarray, n_cols: int = 8) -> plt.Figure:
    """Visualize first n feature maps from a convolutional layer."""
    _style()
    n_maps = min(32, feature_maps.shape[0])
    n_rows = (n_maps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(n_cols * n_rows):
        ax = axes[i]
        if i < n_maps:
            ax.imshow(feature_maps[i], cmap="viridis")
        ax.axis("off")

    fig.suptitle("Convolutional Feature Maps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
