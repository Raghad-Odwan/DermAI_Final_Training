# Final Training Analysis

## 1. Overview
This document presents the analysis of the **final training run** for the DermAI skin lesion classification model.
All reported values are obtained directly from the final training logs and evaluation outputs.

---

## 2. Training Configuration

| Item | Value |
|-----|------|
| Model Architecture | ResNet50 |
| Learning Strategy | Transfer Learning |
| Input Size | 224 × 224 |
| Optimizer | AdamW |
| Loss Function | Binary Focal Cross-Entropy |
| Batch Size | 32 |
| Maximum Epochs | 50 |
| Early Stopping | Enabled (patience = 8) |
| Class Weighting | Applied |

---

## 3. Training Execution Summary

| Metric | Value |
|------|------|
| Total Epochs Trained | 37 |
| Best Epoch | 29 |
| Best Validation Loss | 0.0886 |
| Final Training Accuracy | 81.63% |
| Final Validation Accuracy | 82.77% |
| Total Training Time | 5 hours 16 minutes |

---

## 4. Generalization Behavior

| Metric | Value |
|------|------|
| Train–Validation Accuracy Gap | -1.14% |

**Observation:**  
The small gap between training and validation accuracy indicates stable training behavior without evident overfitting.

---

## 5. Performance Metrics (Threshold = 0.50)

### 5.1 Validation Set

| Metric | Value |
|------|------|
| Accuracy | 83.38% |
| Precision | 73.76% |
| Recall | 74.24% |
| F1-Score | 74.00% |
| AUC-ROC | 91.16% |

---

### 5.2 Test Set

| Metric | Value |
|------|------|
| Accuracy | 83.70% |
| Precision | 74.80% |
| Recall | 73.59% |
| F1-Score | 74.19% |
| AUC-ROC | 91.44% |

---

## 6. Threshold Analysis

### Thresholds Evaluated
```

0.35, 0.40, 0.45, 0.50, 0.55

```

### Optimal Threshold Selection

| Criterion | Value |
|---------|------|
| Selected Threshold | 0.50 |
| Selection Basis | Highest Validation F1-Score |

---

## 7. Confusion Matrix (Test Set)

| | Predicted Benign | Predicted Malignant |
|---|----------------|-------------------|
| Actual Benign | 1,170 | 160 |
| Actual Malignant | 164 | 457 |

---

## 8. Output Artifacts

| Artifact | Description |
|--------|------------|
| final_model_best.keras | Best model checkpoint |
| final_model_complete.keras | Final trained model |
| training_log.csv | Per-epoch training metrics |
| Evaluation Plots | Accuracy, loss, ROC, confusion matrix |

---

## 9. Limitations

| Aspect | Description |
|------|------------|
| Dataset | Single-source dermoscopic dataset |
| Architecture | Single model architecture |
| Evaluation | No external dataset validation |
| Scope | Final training only |

---

## 10. Conclusion
The final training execution resulted in a stable trained model with consistent validation and test performance.
The generated model weights are used directly in the inference and explainability components of the DermAI system.

قولي وأنا أكمّل فورًا.
