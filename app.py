"""app.py - SteelDefect: Steel Surface Defect Detection Dashboard.

A computer vision platform for industrial defect detection with:
- ResNet18 transfer learning vs custom lightweight CNN
- Grad-CAM explainability (visualize what the model "sees")
- Real-time inference benchmarking
- ONNX export for edge deployment
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st
import json

from src.data import (
    load_neu_det, make_synthetic, create_data_splits,
    compute_class_weights, DEFECT_CLASSES, CLASS_ABBREVIATIONS, N_CLASSES
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SteelDefect | Industrial Defect Detection", layout="wide", page_icon="🔍")

# ---------------------------------------------------------------------------
# CSS + Hero
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 55%, #7c3aed 100%);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 { margin-bottom: 0.2rem; }
.hero p  { margin-bottom: 0; opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔍 SteelDefect</h1>
    <p>Computer vision for steel surface defect detection · Grad-CAM explainability · NEU-DET benchmark</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "data" not in st.session_state:
    st.session_state.data = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙ Configuration")
    data_source = st.radio("Dataset", ["NEU-DET (real)", "Synthetic (demo)"], index=0)
    use_synthetic = data_source == "Synthetic (demo)"

    st.divider()
    st.subheader("Training")
    model_choice = st.selectbox("Architecture to train", ["ResNet18 (transfer)", "Custom CNN (lightweight)"])
    pretrained = st.checkbox("Pretrained (ImageNet)", value=True)
    epochs = st.slider("Training epochs", 5, 50, 20)
    lr = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001)
    batch_size = st.slider("Batch size", 8, 64, 32, 8)

    st.divider()
    st.subheader("Benchmark")
    bench_models = st.multiselect(
        "Models to benchmark",
        ["ResNet18", "Custom CNN"],
        default=["ResNet18", "Custom CNN"]
    )

    st.divider()
    st.subheader("Grad-CAM")
    target_class = st.selectbox("Target class", DEFECT_CLASSES)

    st.divider()
    st.caption("Built with PyTorch · Streamlit")
    st.code("streamlit run app.py", language="bash")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data(synthetic: bool):
    if not synthetic:
        data = load_neu_det()
        if data["n_samples"] >= 10:
            return data
    return make_synthetic(n_per_class=50, img_size=64)


data = load_data(use_synthetic)
st.session_state.data = data


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
cols = st.columns(5)
cols[0].metric("Images", f"{data['n_samples']:,}")
cols[1].metric("Classes", data["n_classes"])
cols[2].metric("Image Size", f"{data['img_size']}×{data['img_size']}")
cols[3].metric("Model", model_choice.split()[0])
cols[4].metric("Device", "GPU" if __import__("torch").cuda.is_available() else "CPU")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_train, tab_analysis, tab_gradcam, tab_bench = st.tabs([
    "🔍 Data Explorer", "🧪 Training Lab", "📊 Model Analysis",
    "🎯 Grad-CAM", "⚡ Benchmark"
])


# ===== TAB 1: Data Explorer =====
with tab_data:
    st.subheader("NEU Surface Defect Database")

    from src.visualizations import plot_sample_grid, plot_class_distribution
    st.pyplot(plot_sample_grid(data["images"], data["labels"], data["class_names"]))
    st.pyplot(plot_class_distribution(data["class_counts"], data["class_names"]))

    st.markdown("""
    **About the dataset:**
    - 6 defect types on hot-rolled steel surfaces
    - 300 images per class (balanced)
    - Grayscale images (200×200), converted to RGB for transfer learning
    - Augmentation: random horizontal/vertical flips, rotation ±15°
    """)


# ===== TAB 2: Training Lab =====
with tab_train:
    st.subheader("Model Training")

    if st.button("🚀 Train Model", key="train_btn"):
        import torch
        from src.model import build_resnet18, build_custom_cnn, train_model, evaluate_model

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with st.spinner("Preparing data..."):
            splits = create_data_splits(data["images"], data["labels"])
            class_weights = compute_class_weights(splits["train"]["labels"])

            def make_loader(split_data, shuffle=False):
                from torch.utils.data import DataLoader, TensorDataset
                imgs = torch.FloatTensor(split_data["images"])
                lbls = torch.LongTensor(split_data["labels"])
                return DataLoader(TensorDataset(imgs, lbls), batch_size=batch_size, shuffle=shuffle)

            train_loader = make_loader(splits["train"], shuffle=True)
            val_loader = make_loader(splits["val"])
            test_loader = make_loader(splits["test"])

        with st.spinner(f"Training {model_choice}..."):
            if "ResNet" in model_choice:
                model = build_resnet18(N_CLASSES, pretrained=pretrained)
            else:
                model = build_custom_cnn(N_CLASSES)

            result = train_model(
                model, train_loader, val_loader,
                num_epochs=epochs, lr=lr, device=device, class_weights=class_weights,
            )

        with st.spinner("Evaluating on test set..."):
            eval_result = evaluate_model(result["model"], test_loader, data["class_names"], device=device)

        st.session_state.model = result["model"]
        st.session_state.eval_result = eval_result
        st.session_state.train_history = result["history"]

        st.success(f"Training complete! Test accuracy: {eval_result['accuracy']:.4f}")

        from src.visualizations import plot_training_curves
        st.pyplot(plot_training_curves(result["history"]))


# ===== TAB 3: Model Analysis =====
with tab_analysis:
    st.subheader("Model Evaluation")

    if st.session_state.eval_result is not None:
        eval_res = st.session_state.eval_result

        from src.visualizations import plot_confusion_matrix, plot_per_class_accuracy

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion Matrix**")
            st.pyplot(plot_confusion_matrix(eval_res["confusion_matrix"], data["class_names"]))
        with c2:
            st.markdown("**Per-Class Accuracy**")
            st.pyplot(plot_per_class_accuracy(eval_res["per_class_accuracy"]))

        st.metric("Overall Accuracy", f"{eval_res['accuracy']:.4f}")
    else:
        st.info("Train a model in the Training Lab tab first.")


# ===== TAB 4: Grad-CAM =====
with tab_gradcam:
    st.subheader("Gradient-weighted Class Activation Mapping")

    st.markdown("""
    Grad-CAM visualizes which regions of an image most influenced the model's prediction.

    **Math:** L^c = ReLU(Σ_k α_k · A^k) where α_k = (1/Z) Σ_{i,j} ∂y^c/∂A^k_{ij}

    Red regions contribute most to the predicted class.
    """)

    if st.session_state.model is not None:
        import torch
        from src.model import compute_gradcam
        from src.visualizations import plot_gradcam_overlay

        model = st.session_state.model
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Pick a test image
        test_idx = st.number_input("Test image index", 0, len(data["images"]) - 1, 0)
        target_idx = DEFECT_CLASSES.index(target_class)

        if st.button("Compute Grad-CAM"):
            image = data["images"][test_idx]
            true_label = data["labels"][test_idx]

            with st.spinner("Computing Grad-CAM..."):
                result = compute_gradcam(model, image, target_idx, device=device)

            st.pyplot(plot_gradcam_overlay(
                image, result["heatmap"],
                result["prediction"], result["confidence"],
                true_label=true_label, class_names=data["class_names"],
            ))

            c1, c2, c3 = st.columns(3)
            c1.metric("True Class", data["class_names"][true_label])
            c2.metric("Predicted", data["class_names"][result["prediction"]])
            c3.metric("Confidence", f"{result['confidence']:.1%}")
    else:
        st.info("Train a model first to use Grad-CAM.")


# ===== TAB 5: Benchmark =====
with tab_bench:
    st.subheader("Model Benchmarking")

    st.markdown("""
    Compare models on:
    - **Inference latency** (ms per image)
    - **Model size** (MB)
    - **Parameter count** (millions)
    - **Accuracy** tradeoff

    Critical for edge deployment where compute resources are limited.
    """)

    if not bench_models:
        st.warning("Select at least one model to benchmark.")
    elif st.button("Run Benchmark"):
        import torch
        from src.model import build_resnet18, build_custom_cnn, benchmark_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        splits = create_data_splits(data["images"], data["labels"])

        from torch.utils.data import DataLoader, TensorDataset
        test_imgs = torch.FloatTensor(splits["test"]["images"])
        test_lbls = torch.LongTensor(splits["test"]["labels"])
        test_loader = DataLoader(TensorDataset(test_imgs, test_lbls), batch_size=1)

        benchmarks = {}

        if "ResNet18" in bench_models:
            with st.spinner("Benchmarking ResNet18..."):
                resnet = build_resnet18(N_CLASSES, pretrained=False)
                benchmarks["ResNet18"] = benchmark_model(resnet, test_loader, device)

        if "Custom CNN" in bench_models:
            with st.spinner("Benchmarking Custom CNN..."):
                custom = build_custom_cnn(N_CLASSES)
                benchmarks["Custom CNN"] = benchmark_model(custom, test_loader, device)

        from src.visualizations import plot_model_benchmark
        st.pyplot(plot_model_benchmark(benchmarks))

        # Table
        st.markdown("**Benchmark Results**")
        bench_table = []
        for name, b in benchmarks.items():
            bench_table.append({
                "Model": name,
                "Parameters": f"{b['param_count'] / 1e6:.2f}M",
                "Size (MB)": f"{b['model_size_mb']:.1f}",
                "Latency (ms)": f"{b['avg_inference_ms']:.1f}",
                "Throughput": f"{b['throughput_img_per_sec']:.0f} img/s",
            })
        st.table(bench_table)

        # ONNX export
        st.divider()
        st.subheader("Export to ONNX")
        if st.button("Export ONNX"):
            from src.model import export_onnx
            # benchmark_model() returns metrics, not the model object — export the
            # module we actually benchmarked.
            export_target = resnet if "ResNet18" in benchmarks else custom
            export_onnx(export_target, "models/steel_defect.onnx")
            st.success("Exported to models/steel_defect.onnx")
    else:
        st.info("Click 'Run Benchmark' to compare models.")


# ---------------------------------------------------------------------------
# Deploy notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Deployment & production notes"):
    st.markdown("""
    **SteelDefect** — Edge deployment guide:

    1. **Train** on GPU instance (AWS p3 / GCP T4)
    2. **Export** to ONNX for cross-platform inference
    3. **Quantize** to INT8 for edge devices (Jetson, Raspberry Pi)
    4. **Deploy** with ONNX Runtime or TensorRT
    5. **Monitor** prediction distribution drift in production

    Industrial use cases:
    - Real-time defect detection on production lines
    - Quality grading of steel surfaces
    - Root cause analysis (defect type → process parameter)
    """)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
