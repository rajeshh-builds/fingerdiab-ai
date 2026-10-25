"""
FingerDiab AI — Production Model Training and Benchmarking Pipeline
Trains multiple candidate algorithms with leak-free preprocessing, 5-fold cross validation,
probability calibration, model versioning, and artifact serialization.
"""

import os
import sys
import json
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fingerdiab.ml.preprocessing import build_preprocessing_pipeline, TARGET_CLASS_MAP, FEATURE_COLUMNS
from fingerdiab.ml.calibration import calibrate_model, evaluate_calibration

def train_production_models():
    print("=" * 70)
    print("FINGERDIAB AI — PRODUCTION ML TRAINING & BENCHMARKING SUITE")
    print("=" * 70)
    
    # 1. Dataset Loading
    data_path = os.path.join(BASE_DIR, 'fingerprint_diabetes_dataset.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, 'data', 'raw', 'fingerprint_diabetes_dataset.csv')
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path} with {len(df)} samples and {df.shape[1]} columns.")
    
    # 2. Target and Feature Separation
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df['Diabetes_Risk'].astype(str).str.lower().str.strip()
    y = y_raw.map(TARGET_CLASS_MAP).values
    
    # 3. Leak-Free Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")
    
    # 4. Build and Fit Preprocessing Pipeline ONLY on X_train
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)
    
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    
    # 5. Define Candidate ML Algorithms
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=6, min_samples_split=4, random_state=42
        ),
        'svm': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=42
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5, weights='distance'
        )
    }
    
    benchmark_results = {}
    fitted_models = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nTraining and evaluating candidate algorithms across stratified folds...")
    print("-" * 70)
    
    for name, clf in models.items():
        start_t = time.time()
        
        cv_scores = cross_val_score(clf, X_train_trans, y_train, cv=cv_strategy, scoring='accuracy')
        clf.fit(X_train_trans, y_train)
        train_time = time.time() - start_t
        
        y_pred = clf.predict(X_test_trans)
        y_proba = clf.predict_proba(X_test_trans) if hasattr(clf, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        roc_auc = None
        if y_proba is not None:
            try:
                roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
            except Exception:
                roc_auc = None
                
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        benchmark_results[name] = {
            'accuracy': float(acc),
            'cv_mean_accuracy': float(np.mean(cv_scores)),
            'cv_std_accuracy': float(np.std(cv_scores)),
            'precision_macro': float(prec_macro),
            'recall_macro': float(rec_macro),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'roc_auc_ovr': roc_auc,
            'training_time_sec': round(train_time, 4),
            'confusion_matrix': cm
        }
        fitted_models[name] = clf
        
        print(f"[{name.upper()}] Test Acc: {acc*100:.2f}% | CV Acc: {np.mean(cv_scores)*100:.2f}% (±{np.std(cv_scores)*100:.2f}%) | F1-Macro: {f1_macro:.4f} | Time: {train_time:.3f}s")
        
    primary_clf = fitted_models['random_forest']
    
    print("\nPerforming probability calibration on Random Forest model...")
    calibrated_rf = calibrate_model(primary_clf, X_train_trans, y_train, method='sigmoid', cv=3)
    
    y_prob_uncal = primary_clf.predict_proba(X_test_trans)
    y_prob_cal = calibrated_rf.predict_proba(X_test_trans)
    cal_metrics = evaluate_calibration(y_test, y_prob_uncal, y_prob_cal)
    print(f"Calibration Evaluation: Brier Score Before={cal_metrics['brier_score_before']:.4f}, After={cal_metrics['brier_score_after']:.4f}")
    
    model_dir = os.path.join(BASE_DIR, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    version_id = f"fingerdiab-rf-v1.2.0"
    metadata = {
        'model_name': 'RandomForestClassifier',
        'model_version': version_id,
        'framework': 'scikit-learn',
        'training_timestamp': datetime.now().isoformat(),
        'dataset_file': 'fingerprint_diabetes_dataset.csv',
        'sample_count': len(df),
        'features': FEATURE_COLUMNS,
        'target_classes': ['low', 'medium', 'high'],
        'risk_score_scale': '0.0 - 100.0',
        'benchmark_metrics': benchmark_results,
        'calibration_metrics': cal_metrics,
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 4,
            'random_state': 42
        }
    }
    
    joblib.dump(calibrated_rf, os.path.join(model_dir, 'risk_model.pkl'))
    joblib.dump(pipeline, os.path.join(model_dir, 'pipeline.pkl'))
    
    for name, clf in fitted_models.items():
        joblib.dump(clf, os.path.join(model_dir, f"{name}_model.pkl"))
        
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Training successfully completed! Artifacts saved to '{model_dir}/'")
    print(f"Primary Production Model: risk_model.pkl ({version_id})")
    print("=" * 70)
    return metadata

if __name__ == '__main__':
    train_production_models()
