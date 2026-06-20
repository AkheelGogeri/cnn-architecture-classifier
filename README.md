# Architectural Image Classification — CNN & Transfer Learning

MSc Applied AI | COMP634 Assessment 2 | University of Liverpool (2024/25)
Solo Project

---

## Overview

A comparative study of three neural network approaches for 10-class architectural image classification across 11,534 images. A custom lightweight CNN (TinyDSCNN) was designed from scratch and benchmarked against ResNet18 trained from scratch and ResNet18 with pretrained ImageNet weights. The pretrained model achieved 93.2% test accuracy and 98% top-2 accuracy. The custom TinyDSCNN matched 89.3% accuracy using only 0.15M parameters — 1.3% of ResNet's parameter count.

---

## Dataset

- 11,534 architectural images across 10 classes (dome, column, gargoyle, stained glass, apse, flying buttress, bell tower, altar, vault, arch)
- Images resized to 64×64
- Split: 80% training / 10% validation / 10% test — stratified, fixed seed for reproducibility
- Class imbalance handled via inverse-frequency weighted CrossEntropyLoss

---

## Models

### 1. TinyDSCNN (Custom CNN — from scratch)
A lightweight architecture designed specifically for 64×64 architectural images.

- Convolutional stem → 6 Depthwise-Separable Convolution blocks → Squeeze-and-Excitation modules → Global Average Pooling → Dropout → Linear classifier
- GELU activations throughout
- Depthwise-separable convolutions reduce computation by ~70% vs standard convolutions
- **0.15M parameters** — deliberately under 1M
- Training: AdamW (lr=3e-3, weight_decay=1e-2), OneCycleLR, mixed precision, early stopping (patience=4)

### 2. ResNet18 — Trained from Scratch
- Random weight initialisation, all layers trained
- Tests baseline capacity of ResNet18 without external knowledge
- Training: AdamW (lr=3e-4), CosineAnnealingLR

### 3. ResNet18 — Pretrained + Fine-Tuning
- ImageNet pretrained weights loaded
- Final fully-connected layer replaced for 10-class output
- All layers fine-tuned at lower learning rate (lr=1e-4)
- Strong prior from ImageNet features (edges, curves, textures) transfers well to architectural domains

---

## Results

| Model | Strategy | Params | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|---|---|
| TinyDSCNN | Scratch | 0.150M | 0.881 | 0.893 | 0.881 |
| ResNet18 | Scratch | 11.182M | 0.826 | 0.822 | 0.805 |
| ResNet18 | Pretrained + FT | 11.182M | 0.929 | **0.932** | **0.919** |

### Best Model: ResNet18 (Pretrained + Fine-Tuning)

| Metric | Score |
|---|---|
| Top-1 Accuracy | 0.932 |
| Top-2 Accuracy | 0.980 |
| Macro F1 | 0.919 |
| Weighted F1 | 0.932 |

### Per-Class Highlights
- **Best classes:** stained_glass (F1=0.991), dome_inner (0.971), gargoyle (0.952), column (0.940)
- **Hardest classes:** apse (F1=0.818), flying_buttress (0.820) — high intra-class variation and shared features with adjacent categories

### Key Insight: Efficiency vs Accuracy
TinyDSCNN achieves 89.3% accuracy with 0.15M parameters vs ResNet18's 93.2% with 11.18M parameters. For constrained compute environments, TinyDSCNN offers a compelling trade-off — roughly 74× fewer parameters for only a 3.9% accuracy drop.

---

## Training Components

| Component | TinyDSCNN | ResNet18 |
|---|---|---|
| Optimiser | AdamW | AdamW |
| LR Schedule | OneCycleLR | CosineAnnealingLR |
| Mixed Precision | Yes | Yes |
| Early Stopping | Patience=4 | Patience=4 |
| Loss | Weighted CrossEntropyLoss | Weighted CrossEntropyLoss |

---

## Data Augmentation

**Training:** RandomHorizontalFlip (p=0.5), RandomRotation ±10° (p=0.25), ColorJitter brightness/contrast/saturation 0.1 (p=0.25)

**Validation/Test:** Resize to 64×64 only

All models normalised with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for consistency.

---

## Tech Stack

- Python, PyTorch, torchvision
- Custom Dataset and DataLoader classes
- Mixed precision training (torch.cuda.amp)
- matplotlib, seaborn (visualisation)

---

## Repository Structure

```
cnn-architecture-classifier/
├── notebook.ipynb       # Full implementation: data loading, models, training, evaluation
├── report.pdf           # Written analysis, confusion matrices, discussion
└── README.md
```

---

## Academic Context

- **Module:** COMP634 — Applied AI (Assessment 2)
- **Institution:** University of Liverpool, Department of Computer Science
- **Year:** 2024/25

---

## Author

**Akheel R Gogeri**
MSc Artificial Intelligence with Data Science — University of Liverpool
[LinkedIn](https://linkedin.com/in/akheel-gogeri) | [GitHub](https://github.com/AkheelGogeri)

