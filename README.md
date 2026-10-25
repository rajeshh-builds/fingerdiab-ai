# FingerDiab AI — Biometric Dermatoglyphic & Demographic Diabetes Risk Assessment System

> **Research Prototype Notice**: FingerDiab AI is an academic machine learning research system developed for preliminary, non-invasive risk screening exploration. It does not provide medical diagnoses or replace clinical laboratory tests (HbA1c, OGTT).

---

## 1. Project Overview & Problem Statement
Early identification of diabetes risk is crucial for preventing chronic metabolic microvascular and macrovascular complications. Traditional risk screening relies on invasive venipuncture blood tests or manual lifestyle questionnaires.

**FingerDiab AI** explores non-invasive dermatoglyphic biometric pattern analysis (ridge density, pattern morphology, minutiae structure) combined with foundational demographic and clinical indicators (Age, BMI, Family History) to provide an automated, continuous, calibrated **Diabetes Risk Score (0.0 – 100.0)**.

---

## 2. Key System Capabilities
- **5-Stage Computer Vision Pipeline**: Preprocessing with CLAHE contrast enhancement, Otsu adaptive binarization, Zhang-Suen morphological skeletonization, and minutiae feature extraction.
- **Biometric Quality Assurance Filter**: Pre-screening for blur (Laplacian variance), contrast, brightness, and resolution anomalies before inference.
- **Leak-Free ML Preprocessing**: Scikit-Learn `ColumnTransformer` pipeline fitted strictly on training splits.
- **6-Model Benchmark Suite**: Random Forest, Gradient Boosting, SVM, Decision Tree, KNN, and Logistic Regression with 5-fold cross-validation.
- **Mathematically Calibrated Risk Scoring**: Continuous $0.0 \dots 100.0$ score derived directly from multiclass posterior probabilities $P(\text{Low}), P(\text{Medium}), P(\text{High})$ calibrated via Platt Scaling.
- **Per-Prediction Explainability**: Dynamic feature contribution bars attributing relative factor influence (BMI, Age, Family History, Ridge Count, Pattern).
- **Security & Integrity**: PBKDF2:SHA256 password hashing, SQL injection protection, path traversal defenses, and environment-driven configurations.
- **Automated Test Suite**: 16 unit and integration test suites spanning API, CV, ML inference, and authentication.

---

## 3. Machine Learning Architecture & Benchmark

```
Raw Image ──► Quality Filter ──► CLAHE/Otsu ──► Skeletonization ──► Ridge Extraction ──┐
                                                                                       │
Age, BMI, Family History ─────────────────────────────────────────────────────────────┼──► ColumnTransformer
                                                                                       │
                                                                                       ▼
                                                                             Calibrated RF Model
                                                                                       │
                                                                                       ▼
                                                                           Class Probabilities
                                                                     [P(Low), P(Med), P(High)]
                                                                                       │
                                                                                       ▼
                                                                             Expected Risk Score
                                                                               (0.0 – 100.0)
                                                                                       │
                                                                                       ▼
                                                                            Clinical Dossier &
                                                                          Explainability Report
```

### Benchmark Comparison (Test Set Evaluation)

| Model Name | Test Accuracy | 5-Fold CV Accuracy | F1-Score (Macro) | ROC-AUC (OvR) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Primary)** | **99.50%** | **98.88% (±0.73%)** | **0.9956** | **0.9997** | 0.125 s |
| **Gradient Boosting** | 99.50% | 98.62% (±1.21%) | 0.9956 | 0.9996 | 0.142 s |
| **Decision Tree** | 99.50% | 99.00% (±0.50%) | 0.9956 | 0.9962 | 0.002 s |
| **K-Nearest Neighbors** | 91.00% | 88.88% (±2.42%) | 0.9136 | 0.9782 | 0.007 s |
| **Support Vector Machine (RBF)** | 90.00% | 87.63% (±1.08%) | 0.9051 | 0.9715 | 0.012 s |
| **Logistic Regression** | 77.50% | 78.50% (±1.88%) | 0.7824 | 0.9234 | 0.021 s |

---

## 4. Mathematical Risk Score Derivation
Given posterior probabilities $P(\text{Low})$, $P(\text{Medium})$, and $P(\text{High})$ from the calibrated classifier:

$$\text{Risk Score} = (P_{\text{low}} \times 0.0) + (P_{\text{medium}} \times 50.0) + (P_{\text{high}} \times 100.0)$$

$$\text{Estimated Model Probability} = \frac{\text{Risk Score}}{100.0}$$

### Interpretation Tiers (Model Stratification)
- **0.0 – 33.0 / 100**: Lower Estimated Risk
- **33.1 – 66.0 / 100**: Intermediate Estimated Risk
- **66.1 – 100.0 / 100**: Higher Estimated Risk

---

## 5. REST API Documentation

### Health Check
`GET /api/health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "fingerdiab-rf-v1.2.0",
  "system": "FingerDiab AI"
}
```

### Risk Prediction
`POST /api/predictions/analyze`
```json
// Request Payload:
{
  "age": 45,
  "bmi": 28.4,
  "family_history": "yes",
  "fingerprint_type": "loop",
  "ridge_count": 18,
  "model_choice": "random_forest"
}

// Response:
{
  "success": true,
  "prediction_id": 4,
  "prediction": {
    "risk_score": 67.4,
    "model_probability": 0.674,
    "risk_tier": "rose",
    "risk_interpretation": "Higher Estimated Risk",
    "confidence_percent": 94.8,
    "class_probabilities": {
      "low": 2.1,
      "medium": 30.5,
      "high": 67.4
    }
  },
  "feature_attributions": [
    { "feature": "BMI", "label": "Body Mass Index (BMI)", "percentage": 36.2 },
    { "feature": "Age", "label": "Patient Age", "percentage": 24.5 },
    { "feature": "Family_History", "label": "Genetic Family History", "percentage": 18.3 },
    { "feature": "Ridge_Count", "label": "Dermal Ridge Density", "percentage": 12.0 },
    { "feature": "Fingerprint_Type", "label": "Dermatoglyphic Pattern", "percentage": 9.0 }
  ],
  "model": {
    "name": "Random Forest Classifier",
    "version": "fingerdiab-rf-v1.2.0"
  },
  "disclaimer": "This is an AI-based preliminary risk screening estimate and is not a medical diagnosis."
}
```

---

## 6. How to Run Locally

### 1. Environment Setup
```bash
# Clone or enter directory
cd "C:/Users/miche/Downloads/Code (2)/Code/Code"

# Create and activate virtualenv
python -m venv venv
./venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Validate Dataset & Retrain Models
```bash
python scripts/validate_dataset.py
python scripts/train.py
python scripts/evaluate.py
```

### 3. Run Automated Tests
```bash
python -m unittest discover -s tests -v
```

### 4. Launch Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000/`** in your browser.

---

## 7. Medical & Research Disclaimer
FingerDiab AI is strictly an experimental educational and clinical screening prototype. It does not replace professional medical advice, clinical diagnostics, or blood-based laboratory testing.
