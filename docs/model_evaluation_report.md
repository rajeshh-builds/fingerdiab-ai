# FINGERDIAB AI — Comprehensive Model Evaluation & Benchmark Report

**Model Version**: `fingerdiab-rf-v1.2.0`  
**Training Timestamp**: `2026-10-07T21:50:07.490800`  
**Primary Model**: `RandomForestClassifier` with Platt Scaling Calibration  
**Total Sample Count**: 1000  
**Features Evaluated**: `Fingerprint_Type, Ridge_Count, Age, BMI, Family_History`  

## 1. Candidate Algorithm Benchmark Matrix (5-Fold Stratified CV & Test Split)

| Model Algorithm | Test Accuracy | 5-Fold CV Accuracy | Macro F1-Score | Weighted F1 | Multi-Class ROC-AUC (OvR) | Training Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 99.50% | 98.88% (±0.73%) | 0.9956 | 0.9950 | 1.0000 | 2.094s |
| **Gradient Boosting** | 99.50% | 98.62% (±1.21%) | 0.9956 | 0.9950 | 0.9999 | 5.033s |
| **Logistic Regression** | 77.50% | 78.50% (±1.88%) | 0.7824 | 0.7751 | 0.9315 | 0.135s |
| **Decision Tree** | 99.50% | 99.00% (±0.50%) | 0.9956 | 0.9950 | 0.9964 | 0.034s |
| **Svm** | 90.00% | 87.63% (±1.08%) | 0.9051 | 0.9002 | 0.9802 | 0.586s |
| **Knn** | 91.00% | 88.88% (±2.42%) | 0.9136 | 0.9096 | 0.9787 | 0.059s |

## 2. Probability Calibration Analysis (Platt Scaling via CalibratedClassifierCV)

To ensure continuous numerical risk scores (0-100) reflect true empirical posterior probabilities rather than overconfident uncalibrated ensemble predictions, Platt scaling was applied on 3-fold inner cross validation.

| Metric | Before Calibration | After Calibration | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Brier Score Loss** (Lower is better) | 0.0158 | **0.0068** | **+57.1% error reduction** |
| **Multi-Class Log Loss** (Lower is better) | 0.1456 | **0.0487** | **+66.5% error reduction** |

## 3. Confusion Matrix Breakdown (Primary Random Forest Model on 20% Test Holdout)

| True \ Predicted | Low Risk (Pred) | Medium Risk (Pred) | High Risk (Pred) | Class Total |
|:---|:---:|:---:|:---:|:---:|
| **True Low Risk** | 74 | 0 | 0 | 74 |
| **True Medium Risk** | 1 | 77 | 0 | 78 |
| **True High Risk** | 0 | 0 | 48 | 48 |

## 4. Scientific & Research Grounding
- **Deterministic Inference**: Seed set to `random_state=42` across all candidate models.
- **Leak-Free Guarantee**: Sklearn `ColumnTransformer` fitted strictly on training data.
- **Clinical Screening Distinction**: Evaluated performance metrics reflect the research dataset holdout and do not constitute external clinical validation on diverse hospital cohorts.