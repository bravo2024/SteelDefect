"""model.py - Steel defect classification models with Grad-CAM explainability.

Models implemented:
1. ResNet18 (transfer learning): pretrained on ImageNet, fine-tuned on steel defects
2. Custom CNN: lightweight 3-block architecture for comparison
3. Grad-CAM: gradient-weighted class activation mapping for visual explanations

Mathematical foundations:
- Convolution: (f * g)(t) = Σ f(τ) g(t - τ)
- Batch normalization: y = γ(x - μ) / sqrt(σ² + ε) + β
- Residual connection: y = F(x) + x (identity shortcut)
- Grad-CAM: L^c_Grad-CAM = ReLU(Σ_k α_k · A^k) where α_k = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_{ij}
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time


class ConvBlock:
    """Convolution → BatchNorm → ReLU block.

    Math:
        Conv: y_{ij} = Σ_{k,l} w_{kl} · x_{i+k, j+l} + b
        BN:   ŷ = γ(x̂ - μ_B) / sqrt(σ²_B + ε) + β
        ReLU: f(x) = max(0, x)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        try:
            import torch
            import torch.nn as nn
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.block = nn.Sequential(self.conv, self.bn, self.relu)
        except ImportError:
            raise ImportError("PyTorch required: pip install torch torchvision")

    def __call__(self, x):
        return self.block(x)


class ResidualBlock:
    """Residual block: F(x) + x skip connection.

    Math: y = F(x, {W_i}) + x
    where F is the convolutional function, x is the identity shortcut.
    This enables training of very deep networks by mitigating vanishing gradients.
    """

    def __init__(self, channels: int):
        import torch.nn as nn
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def __call__(self, x):
        import torch
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # skip connection
        out = self.relu(out)
        return out


def build_resnet18(num_classes: int = 6, pretrained: bool = True):
    """Build ResNet18 model for steel defect classification.

    Architecture:
        Input → Conv1 (7×7, stride=2) → BN → ReLU → MaxPool
              → Layer1 (2 × ResBlock, 64 ch) → skip
              → Layer2 (2 × ResBlock, 128 ch, stride=2) → skip
              → Layer3 (2 × ResBlock, 256 ch, stride=2) → skip
              → Layer4 (2 × ResBlock, 512 ch, stride=2) → skip
              → GlobalAvgPool → FC → Softmax

    Math for Global Average Pooling:
        GAP(c) = (1/H·W) Σ_i Σ_j A^c_{ij}
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_custom_cnn(num_classes: int = 6):
    """Custom lightweight CNN for steel defect classification.

    Architecture (3 convolutional blocks):
        Input (3, 224, 224)
        → Conv Block 1: 3→32, 3×3, stride=1, padding=1 → MaxPool 2×2
        → Conv Block 2: 32→64, 3×3, stride=1, padding=1 → MaxPool 2×2
        → Conv Block 3: 64→128, 3×3, stride=1, padding=1 → MaxPool 2×2
        → Flatten → FC 128→64 → ReLU → Dropout(0.5) → FC 64→num_classes

    Total parameters: ~200K (vs 11M for ResNet18)
    """
    import torch
    import torch.nn as nn

    class CustomCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return CustomCNN(num_classes)


def train_model(model, train_loader, val_loader, num_epochs: int = 20,
                lr: float = 0.001, device: str = "cpu", class_weights: Optional[np.ndarray] = None) -> Dict:
    """Train model with cross-entropy loss and Adam optimizer.

    Loss function (weighted cross-entropy):
        L = -Σ_k w_k · y_k · log(p_k)

    Optimizer (Adam):
        m_t = β₁ · m_{t-1} + (1 - β₁) · g_t           (first moment estimate)
        v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          (second moment estimate)
        θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)

    Learning rate scheduler: ReduceLROnPlateau (reduce on validation loss plateau)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

    return {
        "model": model,
        "history": history,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "epochs": num_epochs,
    }


def evaluate_model(model, test_loader, class_names: list, device: str = "cpu") -> Dict:
    """Evaluate model on test set.

    Returns:
        accuracy, per_class_accuracy, confusion_matrix, predictions, true_labels
    """
    import torch
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probas.extend(probas.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)

    accuracy = float(np.mean(all_preds == all_labels))
    cm = sk_confusion_matrix(all_labels, all_preds)

    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float(np.mean(all_preds[mask] == i))
        else:
            per_class_acc[name] = 0.0

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": all_probas,
        "class_names": class_names,
    }


def compute_gradcam(model, image: np.ndarray, target_class: int, device: str = "cpu") -> Dict:
    """Compute Grad-CAM (Gradient-weighted Class Activation Mapping).

    Math:
        1. Forward pass: compute class score y^c = model(x)
        2. Backward pass: compute gradient ∂y^c/∂A^k_{ij} of score w.r.t. feature map A^k
        3. Global average gradient: α_k = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_{ij}
        4. Weighted combination: L^c = ReLU(Σ_k α_k · A^k)

    where:
        A^k is the k-th feature map from the last convolutional layer
        Z = H × W is the spatial size
        ReLU ensures we only highlight regions that positively contribute to the class

    Returns:
        heatmap: np.ndarray of shape (H, W) in [0, 1]
        prediction: predicted class index
        confidence: softmax probability
    """
    import torch

    model.eval()
    img_tensor = torch.FloatTensor(image).unsqueeze(0).to(device)

    # Get feature maps from last conv layer
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks on the last conv layer
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("No Conv2d layer found in model")

    h1 = last_conv.register_forward_hook(forward_hook)
    h2 = last_conv.register_full_backward_hook(backward_hook)

    # Forward + backward
    output = model(img_tensor)
    model.zero_grad()
    target = output[0, target_class]
    target.backward()

    h1.remove()
    h2.remove()

    # Compute Grad-CAM
    feat_map = feature_maps[0].squeeze(0)  # (C, H, W)
    grad = gradients[0].squeeze(0)  # (C, H, W)

    # Global average of gradients
    alpha = grad.mean(dim=(1, 2))  # (C,)

    # Weighted combination
    cam = torch.zeros_like(feat_map[0])
    for k in range(len(alpha)):
        cam += alpha[k] * feat_map[k]

    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to input size
    import torch.nn.functional as F
    cam_resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(image.shape[1], image.shape[2]),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Get prediction
    with torch.no_grad():
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    return {
        "heatmap": cam_resized,
        "prediction": pred_class,
        "confidence": confidence,
        "target_class": target_class,
        "gradients": grad.cpu().numpy(),
        "feature_maps": feat_map.cpu().numpy(),
    }


def benchmark_model(model, test_loader, device: str = "cpu", n_runs: int = 100) -> Dict:
    """Benchmark model inference time and size.

    Metrics:
        - Model size (MB)
        - Average inference time per image (ms)
        - Throughput (images/second)
        - Parameter count
    """
    import torch

    model.eval()

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for b in model.buffers())
    model_size_mb = (param_count * 4 + buffer_count * 4) / (1024 * 1024)  # float32

    # Inference time
    sample_batch = next(iter(test_loader))
    images = sample_batch[0][:1].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(images)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(images)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    avg_time = np.mean(times)
    throughput = 1000.0 / avg_time

    return {
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "avg_inference_ms": float(avg_time),
        "throughput_img_per_sec": float(throughput),
        "n_runs": n_runs,
    }


def export_onnx(model, save_path: str, input_shape: Tuple = (1, 3, 224, 224)):
    """Export PyTorch model to ONNX format."""
    import torch

    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, save_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    return save_path


def quantize_model(model):
    """Apply post-training dynamic quantization.

    Math: Quantize weights from float32 to int8:
        q = round(w / scale + zero_point)
        scale = (w_max - w_min) / 255
        zero_point = round(-w_min / scale)
    """
    import torch

    model_cpu = model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized
