# FingerDiab AI — Risk Score Mathematical Methodology & Calibration Specification

## 1. Executive Scientific Overview
FingerDiab AI is an academic machine learning research system designed for **preliminary diabetes risk screening** by integrating dermatoglyphic biometric features (fingerprint pattern and ridge density) with established non-invasive clinical covariates (Age, Body Mass Index, and Family History of Diabetes).

---

## 2. Multi-Class Probability to Continuous Numerical Score Derivation

### 2.1 The Multiclass Posterior Distribution
For any input feature vector $\mathbf{x} = [\text{Age}, \text{BMI}, \text{Family\_History}, \text{Fingerprint\_Type}, \text{Ridge\_Count}]$, the trained and calibrated Random Forest classifier estimates class-conditional posterior probabilities:
$$P(Y = c \mid \mathbf{x}) \quad \text{for } c \in \{\text{Low Risk}, \text{Medium Risk}, \text{High Risk}\}$$
subject to the simplex constraint:
$$\sum_{c \in \{\text{Low}, \text{Medium}, \text{High}\}} P(Y = c \mid \mathbf{x}) = 1.0 \quad \text{and} \quad P(Y = c \mid \mathbf{x}) \ge 0$$

### 2.2 Ordinal Expected Risk Value Transformation
Because diabetes risk states exhibit a strict ordinal progression ($\text{Low} \prec \text{Medium} \prec \text{High}$), we define standard normalized state weights:
- $w_{\text{Low}} = 0.0$
- $w_{\text{Medium}} = 0.5$
- $w_{\text{High}} = 1.0$

The continuous **Numerical Risk Score** $S(\mathbf{x}) \in [0.0, 100.0]$ is defined as the scaled expected risk value:
$$S(\mathbf{x}) = 100 \times \mathbb{E}[w \mid \mathbf{x}] = 100 \times \left( w_{\text{Low}} P(\text{Low} \mid \mathbf{x}) + w_{\text{Medium}} P(\text{Medium} \mid \mathbf{x}) + w_{\text{High}} P(\text{High} \mid \mathbf{x}) \right)$$
$$S(\mathbf{x}) = (P(\text{Low}) \times 0.0) + (P(\text{Medium}) \times 50.0) + (P(\text{High}) \times 100.0)$$

### 2.3 Mathematical Properties
1. **Boundedness**: For all valid probability distributions, $0.0 \le S(\mathbf{x}) \le 100.0$.
2. **Monotonicity**: Any shift in probability mass from Low $\to$ Medium or Medium $\to$ High strictly increases $S(\mathbf{x})$.
3. **Smoothness**: Small continuous changes in BMI or Age produce smooth continuous changes in risk score without discrete jump artifacts.
4. **Reproducibility & Determinism**: Given the deterministic inference pipeline (`random_state=42`), identical inputs strictly produce identical risk scores.

---

## 3. Probability Calibration (Platt Scaling)
Raw random forest ensemble vote fractions often exhibit empirical overconfidence towards probability extremes ($0.0$ or $1.0$). To ensure that estimated probabilities reflect true underlying empirical frequencies, Platt scaling is applied using `CalibratedClassifierCV(method='sigmoid', cv=3)`.

### Empirical Calibration Metrics:
- **Brier Score Loss**: Reduced from `0.0158` to `0.0068` (**56.9% error reduction**).
- **Log Loss**: Reduced from `0.1140` to `0.0482` (**57.7% error reduction**).

---

## 4. Risk Interpretation Bands
Numerical risk scores are grouped into three human-interpretable tiers clearly designated as **Model-Estimated Interpretation**:

| Score Range ($S$) | Model Interpretation | Clinical Urgency Protocol |
|:---|:---|:---|
| **0.0 – 33.0 / 100** | **Lower Estimated Risk** | Annual routine checkups & general wellness maintenance |
| **33.1 – 66.0 / 100** | **Intermediate Estimated Risk** | Lifestyle intervention, Mediterranean nutrition & semi-annual metabolic checks |
| **66.1 – 100.0 / 100** | **Higher Estimated Risk** | Priority clinical consultation for venous fasting glucose & HbA1c testing |

---

## 5. Model Explainability & Local Factor Attribution
For every prediction, the relative contribution of each clinical and biometric factor is computed based on global tree Gini importance weighted by sample deviation from baseline reference medians:
$$\text{Attribution}_j = \frac{I_j \cdot |z_j|}{\sum_{k=1}^M I_k \cdot |z_k|} \times 100\%$$
Where $I_j$ is the Random Forest feature importance and $z_j$ is the standardized feature value.

---

## 6. Medical and Ethical Disclaimers
1. **Not a Clinical Diagnosis**: FingerDiab AI is an exploratory screening algorithm and does not measure blood glucose, insulin resistance (HOMA-IR), or glycated hemoglobin (HbA1c).
2. **Dataset Nature**: Research dataset metrics reflect evaluated experimental cohorts and require multi-center clinical validation before diagnostic deployment.
