# FingerDiab AI — Technical Architecture & Engineering Specification

## 1. System Topology & Architectural Overview

```
                         ┌─────────────────────────────┐
                         │   Client (Web / Mobile)     │
                         │   HTML5 / TailwindCSS / JS  │
                         └──────────────┬──────────────┘
                                        │ HTTPS / REST
                                        ▼
                         ┌─────────────────────────────┐
                         │   Flask Gateway & Routes    │
                         │  (Auth, Predictions, API)   │
                         └──────┬───────────────┬──────┘
                                │               │
                ┌───────────────┘               └───────────────┐
                ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│   Biometric Vision Engine     │               │   Clinical Inference Engine   │
│  (OpenCV 5-Stage Pipeline)    │               │  (Leak-Free ML Pipeline)      │
│  • Blur & Contrast QA         │               │  • Sklearn ColumnTransformer  │
│  • CLAHE Contrast             │               │  • Calibrated Random Forest   │
│  • Otsu Binarization          │               │  • Expected Risk Derivation   │
│  • Zhang-Suen Skeletonization │               │  • Factor Explainability      │
│  • Minutiae & Ridge Counting  │               └───────────────┬───────────────┘
└───────────────┬───────────────┘                               │
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  Data Layer (SQLite & ORM)  │
                         │  • Parameterized queries    │
                         │  • PBKDF2:SHA256 Auth       │
                         │  • Audited Predictions      │
                         └─────────────────────────────┘
```

## 2. Directory Structure & Module Breakdown
- `fingerdiab/ml/`:
  - `preprocessing.py`: Leak-free sklearn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
  - `calibration.py`: Platt scaling via `CalibratedClassifierCV`.
  - `explainability.py`: Local & global factor attribution calculation.
  - `inference.py`: Singleton model server caching production artifacts.
- `fingerdiab/services/`:
  - `fingerprint_service.py`: 5-stage image processing & base64 visualization pipeline.
  - `risk_service.py`: Mathematical 0-100 risk score calculator.
  - `recommendation_service.py`: Lifestyle and preventive guidance protocols.
- `fingerdiab/database/`:
  - `database.py`: Parameterized connection lifecycle and automatic schema migrations.
  - `repositories.py`: Data access objects (`UserRepository`, `PredictionRepository`).
- `fingerdiab/utils/`:
  - `quality_check.py`: Quantitative image blur, contrast, brightness, and resolution filters.
  - `validators.py`: Strict range and type validation for clinical parameters.
  - `security.py`: Password hashing, path traversal defenses, file extension checks.
- `scripts/`:
  - `train.py`: Full model training and benchmark suite.
  - `evaluate.py`: Generates comprehensive markdown evaluation report.
  - `validate_dataset.py`: Statistical validation of input datasets.
- `docs/`:
  - `risk_score_methodology.md`: Complete mathematical derivation of continuous score.
  - `model_evaluation_report.md`: Benchmark tables and calibration metrics.
  - `dataset_quality_report.md`: Dataset distribution audit.
