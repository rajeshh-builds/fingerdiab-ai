"""
FingerDiab AI — Probability Calibration Module
Evaluates and calibrates machine learning class probabilities using Platt Scaling / Isotonic Regression.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

def calibrate_model(base_estimator, X_train, y_train, method='sigmoid', cv=3):
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv
    )
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf

def evaluate_calibration(y_true, y_prob_uncalibrated, y_prob_calibrated):
    num_classes = y_prob_uncalibrated.shape[1]
    brier_before = []
    brier_after = []
    
    for c in range(num_classes):
        y_binary = (y_true == c).astype(int)
        brier_before.append(brier_score_loss(y_binary, y_prob_uncalibrated[:, c]))
        brier_after.append(brier_score_loss(y_binary, y_prob_calibrated[:, c]))
        
    metrics = {
        'brier_score_before': float(np.mean(brier_before)),
        'brier_score_after': float(np.mean(brier_after)),
        'log_loss_before': float(log_loss(y_true, y_prob_uncalibrated)),
        'log_loss_after': float(log_loss(y_true, y_prob_calibrated)),
        'calibration_improved': bool(np.mean(brier_after) <= np.mean(brier_before))
    }
    return metrics
