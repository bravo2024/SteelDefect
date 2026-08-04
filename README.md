# SteelDefect

Detects surface defects on steel with a CNN, explained via Grad-CAM.

Computer vision platform for detecting 6 defect types (rolled-in scale, patches, crazing, pitted surface, inclusion, scratches) on hot-rolled steel surfaces. Uses the NEU surface defect database with ResNet18 transfer learning and a custom lightweight CNN. Provides data exploration, training lab, model analysis, Grad-CAM heatmap visualisation, and inference benchmarking with ONNX export.

## Run it

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

> **Note:** `train.py` requires PyTorch. The real NEU-DET dataset is **not**
> auto-downloaded — place the six class folders (`RS/`, `Pa/`, `Cr/`, `PS/`,
> `In/`, `Sc/`) under `data/raw/` to train on it. When the folder is absent the
> code transparently falls back to synthetic images for a quick demo.

## Training & explainability

Training is configured with class-weighted cross-entropy loss on 6 defect classes. Grad-CAM overlays highlight regions the model focuses on for each prediction. ONNX export targets edge deployment on production line hardware.

## Dashboard tabs

| Tab | What it does |
|---|---|
| **Data Explorer** | NEU-DET sample grid, class distribution, dataset statistics |
| **Training Lab** | Model architecture selection, hyperparameter config, training curves, confusion matrix |
| **Model Analysis** | Per-class precision/recall, t-SNE feature embeddings, misclassification explorer |
| **Grad-CAM** | Grad-CAM heatmap overlays for model explainability |
| **Benchmark** | Per-image inference time, batch throughput, ONNX vs PyTorch comparison |

## Data

NEU Surface Defect Database: 1,800 grayscale images (200×200), 6 defect classes × 300 per class. Augmented with random flips and rotations. Synthetic fallback (64×64) for demo without download.

### Layout

```
SteelDefect/
  src/         data, model, evaluate, visualizations modules
  train.py     PyTorch training pipeline (ResNet18 + Lightweight CNN)
  app.py       Streamlit dashboard (350 lines)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

Licensed MIT.
