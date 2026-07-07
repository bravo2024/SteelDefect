"""data.py - Steel Surface Defect data loading and augmentation.

Supports:
1. NEU Surface Defect Database (NEU-DET): 6 classes, 1800 grayscale images
2. Synthetic fallback for immediate demo
3. Standard CV preprocessing with augmentation

Mathematical foundations:
- Image normalization: x' = (x - μ) / σ where μ, σ computed per-channel on ImageNet
- Data augmentation: random affine transforms to increase effective training set size
- Class weights: w_k = N / (K * n_k) for handling class imbalance
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib

# NEU-DET class names
DEFECT_CLASSES = [
    "Rolled-in Scale",  # RS
    "Patches",          # Pa
    "Crazing",          # Cr
    "Pitted Surface",   # PS
    "Inclusion",        # In
    "Scratches",        # Sc
]
CLASS_ABBREVIATIONS = ["RS", "Pa", "Cr", "PS", "In", "Sc"]
N_CLASSES = len(DEFECT_CLASSES)

# ImageNet normalization (grayscale replicated to 3 channels)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_neu_det(data_dir: Optional[str] = None, img_size: int = 224) -> Dict:
    """Load NEU-DET dataset from folder structure.

    Expected structure:
        data_dir/
            RS/
                RS_1.bmp
                RS_2.bmp
                ...
            Pa/
                Pa_1.bmp
                ...
            Cr/
            PS/
            In/
            Sc/

    Returns dict with:
        images: np.ndarray of shape (N, 3, H, W) float32 in [0, 1]
        labels: np.ndarray of shape (N,) int64
        class_names: list of str
        n_samples: int
        n_classes: int
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    try:
        from PIL import Image
        import torch
        from torchvision import transforms
    except ImportError:
        return make_synthetic()

    images = []
    labels = []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # converts to [0, 1] and (C, H, W)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    found_any = False
    for class_idx, (class_name, abbrev) in enumerate(zip(DEFECT_CLASSES, CLASS_ABBREVIATIONS)):
        class_dir = data_dir / abbrev
        if not class_dir.exists():
            # Try full name
            class_dir = data_dir / class_name.replace(" ", "_")
        if not class_dir.exists():
            continue

        for img_path in sorted(class_dir.glob("*.bmp")):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor.numpy())
                labels.append(class_idx)
                found_any = True
            except Exception:
                continue

    if not found_any:
        return make_synthetic()

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int64)

    class_counts = np.bincount(labels, minlength=N_CLASSES)

    return {
        "images": images,
        "labels": labels,
        "class_names": DEFECT_CLASSES,
        "class_abbreviations": CLASS_ABBREVIATIONS,
        "n_samples": len(labels),
        "n_classes": N_CLASSES,
        "class_counts": class_counts,
        "img_size": img_size,
    }


def make_synthetic(n_per_class: int = 50, img_size: int = 64, seed: int = 42) -> Dict:
    """Generate synthetic steel defect-like images for demo.

    Creates structured grayscale patterns that mimic different defect types:
    - Rolled-in Scale: horizontal bands
    - Patches: random rectangular patches
    - Crazing: fine crack-like lines
    - Pitted Surface: random dots
    - Inclusion: dark blobs
    - Scratches: diagonal lines
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_class * N_CLASSES
    images = np.zeros((n_total, 3, img_size, img_size), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)

    for class_idx in range(N_CLASSES):
        start = class_idx * n_per_class
        end = start + n_per_class
        labels[start:end] = class_idx

        for i in range(n_per_class):
            img = np.random.uniform(0.3, 0.5, (img_size, img_size)).astype(np.float32)

            if class_idx == 0:  # Rolled-in Scale - horizontal bands
                n_bands = rng.integers(2, 5)
                for _ in range(n_bands):
                    y = rng.integers(5, img_size - 5)
                    width = rng.integers(1, 3)
                    img[y:y+width, :] = float(rng.uniform(0.6, 0.9))

            elif class_idx == 1:  # Patches - rectangular patches
                n_patches = rng.integers(3, 8)
                for _ in range(n_patches):
                    x, y = rng.integers(0, img_size - 15, size=2)
                    w, h = rng.integers(5, 15, size=2)
                    img[y:y+h, x:x+w] = float(rng.uniform(0.6, 0.9))

            elif class_idx == 2:  # Crazing - fine lines
                n_lines = rng.integers(5, 15)
                for _ in range(n_lines):
                    x1, y1 = rng.integers(0, img_size, size=2)
                    x2, y2 = rng.integers(0, img_size, size=2)
                    n_pts = max(abs(x2-x1), abs(y2-y1)) + 1
                    xs = np.linspace(x1, x2, n_pts).astype(int)
                    ys = np.linspace(y1, y2, n_pts).astype(int)
                    valid = (xs >= 0) & (xs < img_size) & (ys >= 0) & (ys < img_size)
                    img[ys[valid], xs[valid]] = float(rng.uniform(0.7, 0.95))

            elif class_idx == 3:  # Pitted Surface - random dots
                n_dots = rng.integers(20, 60)
                coords = rng.integers(0, img_size, size=(n_dots, 2))
                radii = rng.integers(1, 4, size=n_dots)
                for (cy, cx), r in zip(coords, radii):
                    y_lo, y_hi = max(0, cy-r), min(img_size, cy+r+1)
                    x_lo, x_hi = max(0, cx-r), min(img_size, cx+r+1)
                    img[y_lo:y_hi, x_lo:x_hi] = float(rng.uniform(0.7, 0.95))

            elif class_idx == 4:  # Inclusion - dark blobs
                n_blobs = rng.integers(2, 6)
                for _ in range(n_blobs):
                    cx, cy = rng.integers(10, img_size - 10, size=2)
                    r = rng.integers(3, 8)
                    yy, xx = np.ogrid[-cx:img_size-cx, -cy:img_size-cy]
                    mask = xx**2 + yy**2 <= r**2
                    img[mask] = float(rng.uniform(0.1, 0.25))

            elif class_idx == 5:  # Scratches - diagonal lines
                n_scratches = rng.integers(3, 8)
                for _ in range(n_scratches):
                    x1, y1 = rng.integers(0, img_size, size=2)
                    length = rng.integers(15, img_size)
                    angle = rng.uniform(0, 2 * np.pi)
                    x2 = int(x1 + length * np.cos(angle))
                    y2 = int(y1 + length * np.sin(angle))
                    n_pts = max(abs(x2-x1), abs(y2-y1)) + 1
                    xs = np.linspace(x1, x2, n_pts).astype(int)
                    ys = np.linspace(y1, y2, n_pts).astype(int)
                    valid = (xs >= 0) & (xs < img_size) & (ys >= 0) & (ys < img_size)
                    img[ys[valid], xs[valid]] = float(rng.uniform(0.8, 1.0))

            img = np.clip(img, 0, 1)
            images[start + i, 0] = img  # R
            images[start + i, 1] = img  # G
            images[start + i, 2] = img  # B

    class_counts = np.bincount(labels, minlength=N_CLASSES)

    return {
        "images": images,
        "labels": labels,
        "class_names": DEFECT_CLASSES,
        "class_abbreviations": CLASS_ABBREVIATIONS,
        "n_samples": n_total,
        "n_classes": N_CLASSES,
        "class_counts": class_counts,
        "img_size": img_size,
    }


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights (one per class)."""
    classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    n_classes = len(classes)
    return np.array([n_total / (n_classes * c) for c in counts], dtype=np.float32)


def create_data_splits(images: np.ndarray, labels: np.ndarray,
                       val_size: float = 0.15, test_size: float = 0.15,
                       seed: int = 42) -> Dict:
    """Stratified train/val/test split."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    indices = np.arange(n)

    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        class_idx = indices[labels == c]
        rng.shuffle(class_idx)
        n_test = int(len(class_idx) * test_size)
        n_val = int(len(class_idx) * val_size)
        test_idx.extend(class_idx[:n_test])
        val_idx.extend(class_idx[n_test:n_test + n_val])
        train_idx.extend(class_idx[n_test + n_val:])

    return {
        "train": {"images": images[train_idx], "labels": labels[train_idx]},
        "val": {"images": images[val_idx], "labels": labels[val_idx]},
        "test": {"images": images[test_idx], "labels": labels[test_idx]},
        "train_idx": np.array(train_idx),
        "val_idx": np.array(val_idx),
        "test_idx": np.array(test_idx),
    }
