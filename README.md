# Deep Learning-Based Evaluation of Ultrasound Images for Benign Skin Tumors

This project implements the methodology described in the paper **"Deep Learning-Based Evaluation of Ultrasound Images for Benign Skin Tumors"** ([DOI: 10.3390/s23177374](https://doi.org/10.3390/s23177374)). The goal of this project is to classify three types of benign skin tumors (epidermal cyst, lipoma, and pilomatricoma) using a convolutional neural network (CNN) with a combined residual and attention-gated structure.

---

## Introduction

This project is based on the paper's key contributions:
- Development of a combined CNN architecture (Residual + Attention-Gated structure).
- Application of Fast AutoAugment for dataset augmentation.
- Evaluation of the model's performance on ultrasound images of three benign skin tumors.

The trained model achieved the following average metrics:
- **Accuracy:** 95.87%
- **Sensitivity:** 90.10%
- **Specificity:** 96.23%

---

## Dataset

The dataset consists of **698 ultrasound images** collected from patients at Severance Hospital in Seoul, Korea, between November 2017 and January 2020. The dataset includes images of three types of benign skin tumors:
- **Epidermal cyst**: 149 patients
- **Lipoma**: 74 patients
- **Pilomatricoma**: 32 patients

### Preprocessing:
1. **Cropping:** Doppler regions were removed, and only the skin-centered regions were used.
2. **Augmentation:** Fast AutoAugment was applied to increase the dataset by 21 times.

---

## Model Architecture

The CNN model combines two key structures:
1. **Residual Structure**: Based on ResNet18, up to the 4th block.
2. **Attention-Gated Structure**: Highlights important regions in the ultrasound images.

The model outputs probabilities for each of the three tumor classes.

### Training Details:
- **Loss Function:** Focal Loss
- **Optimizer:** Adam
- **Validation:** 5-Fold Cross-Validation
- **Hardware:** NVIDIA RTX 2080Ti GPU

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.11 or higher
- CUDA 10.2 or higher (if using GPU)

### Installation
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/Hyunwoo-vision/USskintumor.git
cd USskintumor
```

---

## Results

The model achieved the following performance metrics across three tumor types:

| Tumor Type       | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|-------------------|--------------|------------------|------------------|
| Epidermal Cyst    | 94.9         | 92.4            | 97.9            |
| Lipoma            | 98.2         | 96.5            | 98.9            |
| Pilomatricoma     | 94.5         | 75.9            | 97.4            |

### Statistical Analysis:
- The model's predictions were compared with those of physicians using CAMs.
- High overlap between model and physician judgment was observed (p-value < 0.001).

---

## License

This project is licensed under the terms of the **Creative Commons Attribution (CC BY)** license.

---
