"""tests/test_smoke.py - Smoke tests for SteelDefect pipeline."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_make_synthetic():
    from src.data import make_synthetic
    data = make_synthetic(n_per_class=10, img_size=32)
    assert data["images"].shape == (60, 3, 32, 32)
    assert data["labels"].shape == (60,)
    assert data["n_classes"] == 6


def test_class_weights():
    from src.data import compute_class_weights
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    weights = compute_class_weights(labels)
    assert weights.shape == (3,)  # one weight per class
    # minority class (0: 3 samples) gets higher weight than majority (2: 4 samples)
    assert weights[0] > weights[2]


def test_create_data_splits():
    from src.data import make_synthetic, create_data_splits
    data = make_synthetic(n_per_class=20, img_size=32)
    splits = create_data_splits(data["images"], data["labels"])
    assert len(splits["train"]["labels"]) > 0
    assert len(splits["val"]["labels"]) > 0
    assert len(splits["test"]["labels"]) > 0


def test_build_custom_cnn():
    import torch
    from src.model import build_custom_cnn
    model = build_custom_cnn(num_classes=6)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 6)


def test_build_resnet18():
    import torch
    from src.model import build_resnet18
    model = build_resnet18(num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)


if __name__ == "__main__":
    test_make_synthetic()
    test_class_weights()
    test_create_data_splits()
    test_build_custom_cnn()
    test_build_resnet18()
    print("All smoke tests passed!")
