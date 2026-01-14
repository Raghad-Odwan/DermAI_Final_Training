# **Final Training Analysis**

## 1. Overview

This document presents a structured academic analysis of the **final training phase** of the DermAI skin lesion classification model.
All reported metrics and observations are **directly derived from the final training logs, evaluation outputs, and saved model artifacts**, ensuring transparency and reproducibility.

The analyzed model represents the **final deployment-ready version** selected after prior experimentation and validation.

---

## 2. Training Configuration

The final training was conducted using the configuration summarized in Table 1. This setup was selected based on prior empirical evaluation and stability considerations.

| Item               | Value                      |
| ------------------ | -------------------------- |
| Model Architecture | ResNet50                   |
| Learning Strategy  | Transfer Learning          |
| Input Size         | 224 × 224                  |
| Optimizer          | AdamW                      |
| Loss Function      | Binary Focal Cross-Entropy |
| Batch Size         | 32                         |
| Maximum Epochs     | 50                         |
| Early Stopping     | Enabled (patience = 8)     |
| Class Weighting    | Applied                    |

---

## 3. Training Execution Summary

Table 2 summarizes the execution statistics of the final training run.

| Metric                       | Value              |
| ---------------------------- | ------------------ |
| Total Epochs Trained         | 37                 |
| Best Epoch (Validation Loss) | 29                 |
| Best Validation Loss         | 0.0886             |
| Final Training Accuracy      | 81.63%             |
| Final Validation Accuracy    | 82.77%             |
| Total Training Time          | 5 hours 16 minutes |

Early stopping was triggered before reaching the maximum number of epochs, indicating convergence and preventing unnecessary overfitting.

---

## 4. Generalization Behavior

To assess generalization, the difference between training and validation accuracy was analyzed.

| Metric                        | Value  |
| ----------------------------- | ------ |
| Train–Validation Accuracy Gap | -1.14% |

**Analysis:**
The validation accuracy slightly exceeds the training accuracy, suggesting **strong generalization behavior** and effective regularization. No signs of overfitting were observed during the training process.

---

## 5. Performance Metrics (Decision Threshold = 0.50)

### 5.1 Validation Set Performance

The model performance on the validation set using a decision threshold of 0.50 is presented below.

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 83.38% |
| Precision | 73.76% |
| Recall    | 74.24% |
| F1-Score  | 74.00% |
| AUC-ROC   | 91.16% |

---

### 5.2 Test Set Performance

The same threshold was applied to the held-out test set to evaluate generalization on unseen data.

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 83.70% |
| Precision | 74.80% |
| Recall    | 73.59% |
| F1-Score  | 74.19% |
| AUC-ROC   | 91.44% |

**Observation:**
The close alignment between validation and test metrics demonstrates consistent performance and confirms the robustness of the trained model.

---

## 6. Threshold Analysis

### 6.1 Evaluated Thresholds

To investigate the effect of the decision threshold on classification performance, the following thresholds were evaluated on the validation set:

```
0.35, 0.40, 0.45, 0.50, 0.55
```

### 6.2 Optimal Threshold Selection

| Criterion          | Value                       |
| ------------------ | --------------------------- |
| Selected Threshold | 0.50                        |
| Selection Basis    | Highest Validation F1-Score |

The selected threshold achieved the best balance between precision and recall without degrading test performance.

---

## 7. Confusion Matrix Analysis (Test Set)

|                  | Predicted Benign | Predicted Malignant |
| ---------------- | ---------------- | ------------------- |
| Actual Benign    | 1,170            | 160                 |
| Actual Malignant | 164              | 457                 |

**Interpretation:**
The confusion matrix indicates a balanced distribution of false positives and false negatives, aligning with the observed precision–recall trade-off and supporting the reliability of the classification outcomes.

---

## 8. Output Artifacts

The following artifacts were generated during the final training and are included to support reproducibility:

| Artifact                     | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `final_model_best.keras`     | Best model checkpoint (lowest validation loss) |
| `final_model_complete.keras` | Final trained model                            |
| `training_log.csv`           | Epoch-wise training metrics                    |
| Evaluation Plots             | Accuracy, loss, ROC curve, confusion matrix    |

---

## 9. Limitations

Despite the strong performance, several limitations should be acknowledged:

| Aspect       | Description                       |
| ------------ | --------------------------------- |
| Dataset      | Single-source dermoscopic dataset |
| Architecture | Single CNN architecture           |
| Evaluation   | No external dataset validation    |
| Scope        | Final training phase only         |

---

## 10. Conclusion

The final training phase of the DermAI model resulted in **stable convergence, strong generalization, and consistent performance across validation and test sets**.
The trained model and its associated artifacts are directly utilized in the inference and explainability components of the DermAI system, making it suitable for deployment within the scope of the graduation project.
