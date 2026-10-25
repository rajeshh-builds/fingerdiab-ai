"""
FingerDiab AI — Probability Calibration Module
Evaluates and calibrates machine learning class probabilities using Platt Scaling / Isotonic Regression.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

def calibrate_model(base_estimator, X_train, y_train, method='sigmoid', cv=3):
    """
    Calibrates model posterior probabilities using Stratified K-Fold cross-validation.
    
    Args:
        base_estimator: Trained or untrained sklearn classifier
        X_train: Preprocessed training features
        y_train: Training labels (0, 1, 2)
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        cv: Cross-validation split count
        
    Returns:
        calibrated_model, calibration_metrics (dict)
    """
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv
    )
    calibrated_clf.fit(X_train, y_train)
    
    return calibrated_clf

def evaluate_calibration(y_true, y_prob_uncalibrated, y_prob_calibrated):
    """
    Computes Brier Score Loss and Log Loss to quantify calibration quality.
    """
    # For multiclass, compute one-vs-rest Brier scores
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
