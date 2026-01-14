# DermAI – Final Model Training

## Overview
This repository contains the implementation of the **final training stage** for the DermAI deep learning system.  
It represents the last training phase after dataset preparation, model comparison, and cross-validation.

The goal of this stage is to train the selected model architecture using the finalized preprocessing pipeline and training configuration, and to produce the final model weights used for inference and explainability.

This repository does not include experimentation or model comparison.

---

## Purpose
The objectives of this repository are:
- Train the selected deep learning model using the full prepared dataset
- Apply finalized preprocessing and augmentation strategies
- Address dataset imbalance using class weighting
- Generate the final trained model weights
- Log training behavior for reproducibility

---

## Model Description
- Architecture: ResNet50 (Transfer Learning)
- Input Size: 224 × 224 RGB images
- Output: Binary classification (Benign / Malignant)
- Loss Function: Binary Cross-Entropy
- Optimization Strategy: Transfer learning with fine-tuning

---

## Repository Structure
```

DermAI_Final_Training/
├── Model/
│   ├── final_model_weights.keras
│   └── training_logs/
│
├── scripts/
│   ├── train_final.py
│   ├── data_generators.py
│   └── train_utils.py
│
├── configs/
│   └── training_config.yaml
│
├── README.md
└── ANALYSIS.md

````

---

## Training Execution
Final training can be executed using the following command:

```bash
python scripts/train_final.py --config configs/training_config.yaml
````

The training configuration file defines:

* Learning rate
* Batch size
* Number of epochs
* Dataset paths
* Augmentation parameters

---

## Output

* Final model weights are saved as `.keras`
* Training history and logs are stored for reference
* The resulting model is used directly in the inference and Grad-CAM modules

احكي وأنا أعدّل فورًا.
