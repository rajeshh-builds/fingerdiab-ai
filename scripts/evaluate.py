"""
FingerDiab AI — Comprehensive Model Evaluation & Benchmark Report Generator
Generates docs/model_evaluation_report.md with detailed statistical tables,
confusion matrices, calibration results, and ROC-AUC comparisons.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'model_metadata.json')
REPORT_PATH = os.path.join(BASE_DIR, 'docs', 'model_evaluation_report.md')

def generate_evaluation_report():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Model metadata not found at {METADATA_PATH}. Run scripts/train.py first.")
        
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
        
    metrics = meta.get('benchmark_metrics', {})
    cal_metrics = meta.get('calibration_metrics', {})
    
    md = []
    md.append("# FINGERDIAB AI — Comprehensive Model Evaluation & Benchmark Report")
    md.append(f"\n**Model Version**: `{meta.get('model_version')}`  ")
    md.append(f"**Training Timestamp**: `{meta.get('training_timestamp')}`  ")
    md.append(f"**Primary Model**: `{meta.get('model_name')}` with Platt Scaling Calibration  ")
    md.append(f"**Total Sample Count**: {meta.get('sample_count')}  ")
    md.append(f"**Features Evaluated**: `{', '.join(meta.get('features', []))}`  \n")
    
    md.append("## 1. Candidate Algorithm Benchmark Matrix (5-Fold Stratified CV & Test Split)\n")
    md.append("| Model Algorithm | Test Accuracy | 5-Fold CV Accuracy | Macro F1-Score | Weighted F1 | Multi-Class ROC-AUC (OvR) | Training Latency |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for name, m in metrics.items():
        disp_name = name.replace('_', ' ').title()
        test_acc = f"{m['accuracy']*100:.2f}%"
        cv_acc = f"{m['cv_mean_accuracy']*100:.2f}% (±{m['cv_std_accuracy']*100:.2f}%)"
        f1_m = f"{m['f1_macro']:.4f}"
        f1_w = f"{m['f1_weighted']:.4f}"
        auc_str = f"{m['roc_auc_ovr']:.4f}" if m.get('roc_auc_ovr') is not None else "N/A"
        lat = f"{m['training_time_sec']:.3f}s"
        md.append(f"| **{disp_name}** | {test_acc} | {cv_acc} | {f1_m} | {f1_w} | {auc_str} | {lat} |")
        
    md.append("\n## 2. Probability Calibration Analysis (Platt Scaling via CalibratedClassifierCV)\n")
    md.append("To ensure continuous numerical risk scores (0-100) reflect true empirical posterior probabilities rather than overconfident uncalibrated ensemble predictions, Platt scaling was applied on 3-fold inner cross validation.\n")
    md.append("| Metric | Before Calibration | After Calibration | Relative Improvement |")
    md.append("|:---|:---:|:---:|:---:|")
    brier_bef = cal_metrics.get('brier_score_before', 0)
    brier_aft = cal_metrics.get('brier_score_after', 0)
    brier_imp = ((brier_bef - brier_aft) / max(1e-6, brier_bef)) * 100
    
    ll_bef = cal_metrics.get('log_loss_before', 0)
    ll_aft = cal_metrics.get('log_loss_after', 0)
    ll_imp = ((ll_bef - ll_aft) / max(1e-6, ll_bef)) * 100
    
    md.append(f"| **Brier Score Loss** (Lower is better) | {brier_bef:.4f} | **{brier_aft:.4f}** | **+{brier_imp:.1f}% error reduction** |")
    md.append(f"| **Multi-Class Log Loss** (Lower is better) | {ll_bef:.4f} | **{ll_aft:.4f}** | **+{ll_imp:.1f}% error reduction** |")
    
    md.append("\n## 3. Confusion Matrix Breakdown (Primary Random Forest Model on 20% Test Holdout)\n")
    rf_cm = metrics.get('random_forest', {}).get('confusion_matrix', [])
    if rf_cm:
        md.append("| True \\ Predicted | Low Risk (Pred) | Medium Risk (Pred) | High Risk (Pred) | Class Total |")
        md.append("|:---|:---:|:---:|:---:|:---:|")
        class_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        for idx, row in enumerate(rf_cm):
            md.append(f"| **True {class_labels[idx]}** | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
            
    md.append("\n## 4. Scientific & Research Grounding")
    md.append("- **Deterministic Inference**: Seed set to `random_state=42` across all candidate models.")
    md.append("- **Leak-Free Guarantee**: Sklearn `ColumnTransformer` fitted strictly on training data.")
    md.append("- **Clinical Screening Distinction**: Evaluated performance metrics reflect the research dataset holdout and do not constitute external clinical validation on diverse hospital cohorts.")
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(md))
        
    print(f"Evaluation report generated successfully at {REPORT_PATH}")

if __name__ == '__main__':
    generate_evaluation_report()
