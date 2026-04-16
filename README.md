# Cross-Dataset Transfer and Unknown-Class Detection in SAR Ship Classification

This repository contains the code and experiment notebook accompanying the paper:

> **Cross-Dataset Transfer and Unknown-Class Detection in SAR Ship Classification**

---

## Abstract

Synthetic aperture radar (SAR) models are often trained on one dataset and deployed on another, but their robustness under this shift is unclear. We evaluate six pretrained models on two SAR ship datasets in three settings: **in-domain classification**, **cross-dataset transfer**, and **unknown-class detection**. For unknown detection, we hold out all classes one at a time.

- **In-domain**: SARDet100K gives the best balanced accuracy on both datasets (73.4% on FUSARShip and 53.2% on OpenSARShip).
- **Cross-dataset transfer**: Strong failures are observed — some models show moderate accuracy but near-chance balanced accuracy (e.g., 64.5% accuracy but 33.3% balanced accuracy for OpenSARShip → FUSARShip).
- **Unknown-class detection**: Performance depends on the held-out class and dataset, while MC-dropout variance is often close to random.

These findings show that cross-dataset generalization in SAR remains limited and that task-specific uncertainty scores are often more informative than MC-dropout variance for held-out-class detection, although their relative ranking depends on the dataset and held-out class.

---

## Repository Contents

| File | Description |
|------|-------------|
| `experiment notebook.ipynb` | End-to-end experiment notebook (data preparation → training → evaluation → analysis) |

---

## Experiment Notebook

The companion notebook `experiment notebook.ipynb` reproduces all experiments in the paper. It is structured as a self-contained Colab notebook and covers the following stages:

### 1. Data Preparation

Embeddings for both datasets are loaded from pre-extracted `.npz` files. The notebook combines the original train/val/test pools, filters to the three paper classes (Cargo, Tanker, Fishing), and constructs **5 stratified random splits** (80 / 10 / 10) to reduce sensitivity to a single split. Index files and metadata are saved for reproducibility.

### 2. Model Backbones

Six Earth Observation foundation models are benchmarked. Each backbone is **frozen** — only a lightweight two-layer MLP classifier head is trained on top of the cached embeddings.

| Model | Description |
|-------|-------------|
| [DOFA](https://github.com/zhu-xlab/DOFA) | Multimodal EO model for varying sensor types |
| [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) | Transformer pretrained on HLS imagery |
| [ScaleMAE](https://github.com/bair-climate-initiative/scale-mae) | Scale-aware remote-sensing features |
| [SSL4EO](https://github.com/zhu-xlab/SSL4EO-S12) | Self-supervised weights from SSL4EO-S12 |
| [SARDet100K](https://github.com/zcablii/sardet_100k) | Pretrained on SAR object detection |
| [SARJEPA](https://github.com/waterdisappear/SAR-JEPA) | SAR joint-embedding predictive learning |

### 3. Training

The classifier head is trained with cross-entropy loss (Adam, lr = 1e-3) for up to 100 epochs with early stopping (patience = 20). Experiments are repeated over **5 data splits × 5 random seeds** for robustness.

### 4. Evaluation

The notebook evaluates models under three protocols:

- **In-Domain (ID):** trained and tested on the same dataset.
- **Cross-Dataset (CD):** trained on one dataset, tested on the other (both directions).
- **Unknown-Class (UC):** one class is withheld from training; AUROC measures how well each uncertainty score flags those samples at test time.

---

## Datasets

| Dataset | Sensor | Description |
|---------|--------|-------------|
| [FUSARShip](https://emwlab.fudan.edu.cn/67/05/c20899a222981/page.htm) | GF-3 | High-resolution SAR-AIS matchups |
| [OpenSARShip](https://opensar.sjtu.edu.cn/project.html) | Sentinel-1 | Ship chips linked to AIS messages |

Both datasets are restricted to the **shared ship categories** (Cargo, Tanker, Fishing). All images are resized to 224 × 224 and normalized to [0, 1].

---

## Uncertainty Methods

Four scores are computed and compared for unknown-class detection:

| Method | Formula |
|--------|---------|
| Softmax | $1 - \max_k p_k(x)$ |
| MC-Dropout Variance | Mean per-class variance over 10 stochastic forward passes |
| Class-Oriented ($s_\text{class}$) | Min Euclidean distance to any class prototype |
| Domain-Aware ($s_\text{domain}$) | Euclidean distance to the global training centroid |

---

## Key Results

| Setting | Highlight |
|---------|-----------|
| In-domain | SARDet100K: 73.4% balanced accuracy on FUSARShip, 53.2% on OpenSARShip |
| Cross-dataset | OpenSARShip → FUSARShip: 64.5% accuracy but only 33.3% balanced accuracy |
| Unknown-class | Softmax / class-oriented scores outperform MC-dropout variance on most experiments |

---

## Citation

If you use this work, please cite:

```bibtex
@article{sar_uncertainty_2024,
  title   = {Cross-Dataset Transfer and Unknown-Class Detection in SAR Ship Classification},
  year    = {2024}
}
```

---

## License

This repository is provided for research purposes. Please refer to the individual dataset and model licenses for terms of use.
