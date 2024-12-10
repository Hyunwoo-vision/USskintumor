# Deep Learning-Based Evaluation of Ultrasound Images for Benign Skin Tumors

## Overview
This project implements a **deep learning-based diagnostic model** for **benign skin tumors** using ultrasound images. The model combines **Residual Structures** and **Attention-Gated Structures** to classify three types of benign tumors:  
- **Epidermal Cyst**  
- **Lipoma**  
- **Pilomatricoma**  

The study achieved high diagnostic accuracy and demonstrated that the model's decision-making aligns closely with physicians' criteria.

---

## Key Features
- **Combined CNN Architecture**: Integrates Residual and Attention-Gated structures for enhanced performance.  
- **Data Augmentation**: Utilizes the **Fast AutoAugment** technique to improve model robustness with limited data.  
- **Class Activation Map (CAM)**: Visualizes the regions of interest used by the model for classification.  
- **High Performance**: Achieved an average accuracy of over 94% for all tumor types.  

---

## Dataset
- **Source**: Ultrasound images collected from Severance Hospital, Seoul, Korea.  
- **Size**: 698 images from 250 patients.  
- **Classes**:  
  - Epidermal Cyst (149 cases)  
  - Lipoma (74 cases)  
  - Pilomatricoma (32 cases)  

---

## Model Details
- **Architecture**:  
  - Residual Structure: Based on ResNet18.  
  - Attention-Gated Structure: Enhances focus on critical regions in ultrasound images.  
- **Training**:  
  - 5-Fold Cross-Validation.  
  - Loss Function: Focal Loss.  
  - Optimizer: Adam.  
- **Environment**:  
  - Python 3.9.5  
  - PyTorch 1.7.1  
  - CUDA 10.1  

---

## Results
- **Performance Metrics**:  
  - **Epidermal Cyst**: Accuracy 94.9%, Sensitivity 92.4%, Specificity 97.9%.  
  - **Lipoma**: Accuracy 98.2%, Sensitivity 96.5%, Specificity 98.9%.  
  - **Pilomatricoma**: Accuracy 94.5%, Sensitivity 75.9%, Specificity 97.4%.  
- **Statistical Validation**: High correlation between model predictions and physicians' judgments (p-value < 0.001).  

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/benign-skin-tumor-diagnosis.git
   cd benign-skin-tumor-diagnosis
