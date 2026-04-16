# Cross-Dataset Transfer and Unknown-Class Detection in SAR Ship Classification

## Abstract

Synthetic aperture radar (SAR) models are often trained on one dataset and deployed on another, but their robustness under this shift is unclear. We evaluate six pretrained models on two SAR ship datasets in three settings: **in-domain classification**, **cross-dataset transfer**, and **unknown-class detection**. For unknown detection, we hold out all classes one at a time.

- **In-domain**: SARDet100K gives the best balanced accuracy on both datasets (73.4% on FUSARShip and 53.2% on OpenSARShip).
- **Cross-dataset transfer**: Strong failures are observed — some models show moderate accuracy but near-chance balanced accuracy (e.g., 64.5% accuracy but 33.3% balanced accuracy for OpenSARShip → FUSARShip).
- **Unknown-class detection**: Performance depends on the held-out class and dataset, while MC-dropout variance is often close to random.

These findings show that cross-dataset generalization in SAR remains limited and that task-specific uncertainty scores are often more informative than MC-dropout variance for held-out-class detection, although their relative ranking depends on the dataset and held-out class.

---

## Methodology

### Overview

The study pipeline evaluates SAR foundation-model representations by freezing each backbone and training a lightweight classifier head under a fixed protocol.

### Foundation Models

We benchmark six Earth Observation (EO) foundation models:

| Model | Description |
|-------|-------------|
| [DOFA](https://github.com/zhu-xlab/DOFA) | Multimodal Earth observation model designed for varying sensor types and channel configurations |
| [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) | Transformer-based geospatial model pretrained on HLS imagery |
| [ScaleMAE](https://github.com/bair-climate-initiative/scale-mae) | Scale-aware remote-sensing features via masked autoencoding |
| [SSL4EO](https://github.com/zhu-xlab/SSL4EO-S12) | Self-supervised weights trained on the SSL4EO-S12 Sentinel-1/Sentinel-2 corpus |
| [SARDet100K](https://github.com/zcablii/sardet_100k) | Pretrained on large-scale SAR object detection |
| [SARJEPA](https://github.com/waterdisappear/SAR-JEPA) | SAR self-supervised model based on joint-embedding predictive learning |

### Datasets

We evaluate on two public SAR ship datasets:

| Dataset | Sensor | Description |
|---------|--------|-------------|
| [FUSARShip](https://emwlab.fudan.edu.cn/67/05/c20899a222981/page.htm) | GF-3 | High-resolution SAR-AIS matchups |
| [OpenSARShip](https://opensar.sjtu.edu.cn/project.html) | Sentinel-1 | Ship chips linked to AIS messages |

Both datasets are restricted to their **shared ship categories** (Cargo, Tanker, Fishing) to ensure comparability.

**Preprocessing:**
- All images resized to 224×224 pixels
- Normalized to [0, 1]
- SAR channels replicated as needed to satisfy backbone input requirements
- Dataset splits: 80/10/10 train/validation/test (stratified by class)

### Classification Pipeline

For a sample $(x, y)$, where $x$ is a SAR image and $y \in \{\text{Cargo}, \text{Tanker}, \text{Fishing}\}$:

1. Frozen backbone $\phi_m(\cdot)$ extracts feature vector $z = \phi_m(x)$
2. Classifier head $h_\theta(\cdot)$ maps $z$ to class logits

**Classifier head:** Two-layer MLP with hidden dimension 256, ReLU activation, and dropout probability 0.2.

**Training configuration:**
- Loss: Cross-entropy
- Optimizer: Adam (lr = 1e-3, weight decay = 1e-4)
- Batch size: 512
- Max epochs: 100
- Early stopping: patience = 20 (monitoring validation loss)
- Repeated over **5 data splits × 5 random seeds**

### Uncertainty Scores

Four uncertainty quantification methods are evaluated:

#### 1. Maximum Softmax Probability (Softmax)

$$u_{\text{soft}}(x) = 1 - \max_k p_k(x)$$

Higher values indicate lower model confidence.

#### 2. MC-Dropout Variance (MC-Var)

With dropout enabled in the classifier during inference, $T = 10$ stochastic forward passes are performed on the cached embedding $z$:

$$u_{\text{mc}}(x) = \frac{1}{K} \sum_{k=1}^K \mathrm{Var}_t\!\left(p_k^{(t)}(x)\right)$$

where $K = 3$ is the number of classes. Higher values indicate greater prediction variability.

#### 3. Class-Oriented Uncertainty ($s_{\text{class}}$)

Class prototypes are computed as mean feature vectors per class from the source training set:

$$\mu_c = \frac{1}{|D_c|} \sum_{(x_i, y_i) \in D_c} \phi_m(x_i), \quad c \in \{1, 2, 3\}$$

Uncertainty is the minimum Euclidean distance to any prototype:

$$s_{\text{class}}(x) = \min_{c \in \{1,2,3\}} \left\lVert z - \mu_c \right\rVert_2$$

#### 4. Domain-Aware Uncertainty ($s_{\text{domain}}$)

A global domain center is computed as the mean of all training features:

$$\mu_{\text{dom}} = \frac{1}{|D_{\text{train}}|} \sum_{(x_i, y_i) \in D_{\text{train}}} \phi_m(x_i)$$

Domain-aware uncertainty is the Euclidean distance to this center:

$$s_{\text{domain}}(x) = \left\lVert z - \mu_{\text{dom}} \right\rVert_2$$

All four scores are normalized so that larger values indicate higher uncertainty.

---

## Evaluation Protocol

### 1. In-Domain (ID) Evaluation

Models are trained and tested on the **same dataset** using stratified five-fold cross-validation. This setting measures supervised performance under distribution match.

### 2. Cross-Dataset (CD) Evaluation

Transfer learning is evaluated by training on one dataset and testing on the other, in both directions:
- FUSARShip → OpenSARShip
- OpenSARShip → FUSARShip

This measures robustness to dataset shift while maintaining known-class structure.

### 3. Unknown-Class (UC) Evaluation

Out-of-distribution detection is simulated by holding out each class during training:

For each class $c \in \{\text{Cargo}, \text{Tanker}, \text{Fishing}\}$:
1. Remove all samples with label $c$ from the training set
2. Train the classifier head on the two remaining known classes
3. At test time, evaluate on the full test set (all three classes)
4. Treat class $c$ as unknown

Uncertainty scores are computed for all test samples, and unknown-detection performance is evaluated via **AUROC**.

This procedure is repeated three times (once per held-out class), yielding three independent unknown-class experiments.

---

## Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Balanced Accuracy | Co-primary metric under class imbalance (alongside Macro-F1) |
| Macro-F1 | Co-primary metric under class imbalance (alongside Balanced Accuracy) |
| Per-class Recall | Per-class performance breakdown |
| AUROC | Threshold-free metric for unknown-class detection (0.5 = random, 1.0 = perfect) |

---

## Key Results

- **In-domain**: SARDet100K achieves the best balanced accuracy on both datasets (73.4% on FUSARShip; 53.2% on OpenSARShip).
- **Cross-dataset**: Significant performance degradation is observed. Example: 64.5% accuracy but only 33.3% balanced accuracy for OpenSARShip → FUSARShip transfer.
- **Unknown-class detection**: Task-specific uncertainty scores (softmax, class-oriented, domain-aware) are often more informative than MC-dropout variance for detecting held-out classes.

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
