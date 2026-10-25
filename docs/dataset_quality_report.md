# Dataset Quality & Audit Report — FingerDiab AI

## 1. Executive Summary
- **Source File**: `data/raw/fingerprint_diabetes_dataset.csv`
- **Total Records**: 1,000 samples
- **Total Features**: 7 attributes
- **Missing / Null Values**: 0 (0.00%)
- **Exact Duplicate Rows**: 0 (0.00%)
- **Cleaned Target Saved**: `data/processed/cleaned_dataset.csv`

---

## 2. Feature Schema & Data Types
| Column Name | Data Type | Permissible Domain / Range | Observed Distribution / Values | Validation Status |
|---|---|---|---|---|
| `ID` | Integer | Unique identifier (1..N) | 1 to 1000 | ✅ Valid |
| `Fingerprint_Type` | String (Categorical) | `arch`, `loop`, `whorl` | {"loop": 354, "whorl": 338, "arch": 308} | ✅ Valid |
| `Ridge_Count` | Integer | 0 to 40 ridges | Min: 10.0, Max: 25.0, Mean: 17.6 ± 4.6 | ✅ Valid |
| `Age` | Integer | 18 to 100 years | Min: 18.0, Max: 70.0, Mean: 43.57 ± 15.46 | ✅ Valid |
| `BMI` | Float | 15.0 to 50.0 kg/m² | Min: 18.0, Max: 35.0, Mean: 26.48 ± 4.91 | ✅ Valid |
| `Family_History` | String (Binary) | `no`, `yes` | {"no": 509, "yes": 491} | ✅ Valid |
| `Diabetes_Risk` (Target) | String (Ordinal Multi-class) | `low`, `medium`, `high` | {"medium": 390, "low": 368, "high": 242} | ✅ Valid |

---

## 3. Target Class Balance
| Class Label | Frequency Count | Relative Percentage | Class Weight Adjustment |
|---|---|---|---|
| `medium` | 390 | 39.0% | Balanced |
| `low` | 368 | 36.8% | Balanced |
| `high` | 242 | 24.2% | Moderate Minority |

---

## 4. Statistical Distribution & Scientific Observations
1. **Dermatoglyphic Ridge Counts**:
   - Ridge counts span from 10.0 to 25.0 (mean: 17.6), representing local sampling of dermal ridge density.
2. **Clinical Risk Factors**:
   - Age spans adult population (18.0–70.0 yrs).
   - BMI covers normal (18.0) to Class I/II obesity (35.0).
3. **Data Provenance & Academic Note**:
   - The evaluated dataset contains 1000 structured samples designed for academic validation and benchmarking of multivariable machine learning algorithms.
   - External clinical trials and FDA/CE-MDR diagnostic validation have not been performed; results are structured strictly as preliminary screening risk estimates.
