"""train.py - Steel defect detection training pipeline."""
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import argparse
import pickle
import json
from src.data import load_neu_det, make_synthetic, create_data_splits, compute_class_weights
from src.model import build_resnet18, build_custom_cnn, train_model, evaluate_model


def make_torch_loaders(splits: Dict, batch_size: int = 32):
    """Convert numpy arrays to PyTorch DataLoaders."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def to_loader(data, shuffle=False):
        images = torch.FloatTensor(data["images"])
        labels = torch.LongTensor(data["labels"])
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return {
        "train": to_loader(splits["train"], shuffle=True),
        "val": to_loader(splits["val"]),
        "test": to_loader(splits["test"]),
    }


def main():
    parser = argparse.ArgumentParser(description="SteelDefect training pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--model", choices=["resnet18", "custom_cnn"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    try:
        if not args.synthetic:
            data = load_neu_det()
            if data["n_samples"] < 10:
                raise ValueError("Not enough images found")
            print(f"Loaded NEU-DET: {data['n_samples']} images, {data['n_classes']} classes")
        else:
            raise FileNotFoundError
    except Exception:
        print("NEU-DET not found. Using synthetic data.")
        data = make_synthetic(n_per_class=50, img_size=64)

    # Split
    splits = create_data_splits(data["images"], data["labels"], seed=args.seed)
    print(f"Split: train={len(splits['train']['labels'])}, val={len(splits['val']['labels'])}, test={len(splits['test']['labels'])}")

    # Class weights
    class_weights = compute_class_weights(splits["train"]["labels"])

    # Build model
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.model == "resnet18":
        model = build_resnet18(num_classes=data["n_classes"], pretrained=True)
    else:
        model = build_custom_cnn(num_classes=data["n_classes"])

    # Train
    loaders = make_torch_loaders(splits, batch_size=32)
    print(f"\nTraining {args.model} for {args.epochs} epochs...")
    result = train_model(
        model, loaders["train"], loaders["val"],
        num_epochs=args.epochs, lr=args.lr, device=device,
        class_weights=class_weights,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    eval_result = evaluate_model(result["model"], loaders["test"], data["class_names"], device=device)
    print(f"Test Accuracy: {eval_result['accuracy']:.4f}")
    for name, acc in eval_result["per_class_accuracy"].items():
        print(f"  {name}: {acc:.4f}")

    # Save
    import torch
    Path("models").mkdir(exist_ok=True)
    torch.save(result["model"].state_dict(), "models/model.pt")
    with open("models/metrics.json", "w") as f:
        json.dump({
            "accuracy": eval_result["accuracy"],
            "per_class_accuracy": eval_result["per_class_accuracy"],
            "history": result["history"],
        }, f, indent=2)
    print("\nSaved model -> models/model.pt")


if __name__ == "__main__":
    main()
