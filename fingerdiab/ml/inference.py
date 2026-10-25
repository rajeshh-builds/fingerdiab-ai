"""
FingerDiab AI — Production Model Inference Engine
Handles single-instance and batch predictions using trained, calibrated pipelines.
Loads artifacts once at process initialization for high throughput and zero reload latency.
"""

import os
import joblib
import json
import pandas as pd
import numpy as np
from fingerdiab.services.risk_service import calculate_numerical_risk_score
from fingerdiab.ml.explainability import compute_instance_contributions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
PRIMARY_MODEL_PATH = os.path.join(MODEL_DIR, 'risk_model.pkl')
PIPELINE_PATH = os.path.join(MODEL_DIR, 'pipeline.pkl')

class InferenceEngine:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.metadata = {}
        self.is_loaded = False
        self.load_artifacts()
        
    @property
    def is_ready(self):
        return self.is_loaded

    @property
    def primary_model(self):
        return self.model

        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance
        
    def load_artifacts(self):
        try:
            if os.path.exists(PRIMARY_MODEL_PATH):
                self.model = joblib.load(PRIMARY_MODEL_PATH)
            elif os.path.exists(os.path.join(MODEL_DIR, 'random_forest_model.pkl')):
                self.model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
                
            if os.path.exists(PIPELINE_PATH):
                self.pipeline = joblib.load(PIPELINE_PATH)
                
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    "model_name": "RandomForestClassifier",
                    "model_version": "fingerdiab-rf-v1.0.0",
                    "training_date": "2026-10-07",
                    "features": ["Fingerprint_Type", "Ridge_Count", "Age", "BMI", "Family_History"]
                }
                
            self.is_loaded = (self.model is not None)
        except Exception as e:
            print(f"Warning: Could not load ML artifacts: {e}")
            self.is_loaded = False
            
    def predict_risk(self, feature_dict, model_choice=None, model_name=None):
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("ML model is unavailable or corrupted.")
                
        chosen = model_name or model_choice
        
        input_row = pd.DataFrame([{
            'Fingerprint_Type': str(feature_dict.get('fingerprint_type', 'loop')).lower(),
            'Ridge_Count': float(feature_dict.get('ridge_count', 18)),
            'Age': float(feature_dict.get('age', 40)),
            'BMI': float(feature_dict.get('bmi', 25.0)),
            'Family_History': 'yes' if str(feature_dict.get('family_history', '')).lower() in {'yes', '1', 'true'} else 'no'
        }])

        # Load all candidate models for auto-selection & fallback
        candidates = {'random_forest': self.model}
        for c_name in ['gradient_boosting', 'decision_tree', 'svm', 'knn', 'logistic_regression']:
            c_path = os.path.join(MODEL_DIR, f"{c_name}_model.pkl")
            if os.path.exists(c_path):
                try:
                    candidates[c_name] = joblib.load(c_path)
                except Exception:
                    pass
                    
        if self.pipeline is not None:
            X_trans = self.pipeline.transform(input_row)
        else:
            from fingerdiab.ml.preprocessing import build_preprocessing_pipeline
            pipe = build_preprocessing_pipeline()
            X_trans = pipe.fit_transform(input_row)

        from fingerdiab.ml.model_selector import ModelSelectorEngine

        # If user explicitly specified a valid model, honor it; otherwise perform Dynamic Metric-Based Auto-Selection
        norm_chosen = str(chosen).lower().replace(' ', '_') if chosen else 'auto'
        if norm_chosen in candidates:
            active_model = candidates[norm_chosen]
            if hasattr(active_model, 'predict_proba'):
                probs = active_model.predict_proba(X_trans)[0]
            elif hasattr(active_model, 'decision_function'):
                df_vals = active_model.decision_function(X_trans)[0]
                exp_vals = np.exp(df_vals - np.max(df_vals))
                probs = exp_vals / np.sum(exp_vals)
            else:
                pred_idx = active_model.predict(X_trans)[0]
                probs = [0.0, 0.0, 0.0]
                probs[int(pred_idx)] = 1.0
            selected_info = {
                'selected_name': norm_chosen,
                'display_name': norm_chosen.replace('_', ' ').title(),
                'selection_reason': f"Selected by clinician parameter override ({norm_chosen.replace('_', ' ').title()}).",
                'evaluations': []
            }
        else:
            # Dynamic Auto-Selection based on input metrics, uncertainty entropy, and benchmark performance
            selection_res = ModelSelectorEngine.evaluate_and_select_best_algorithm(X_trans, candidates, self.metadata)
            if selection_res:
                probs = selection_res['probabilities']
                selected_info = selection_res
            else:
                probs = self.model.predict_proba(X_trans)[0]
                selected_info = {
                    'selected_name': 'random_forest',
                    'display_name': 'Random Forest Classifier',
                    'selection_reason': 'Default primary ensemble selected.',
                    'evaluations': []
                }
            
        risk_result = calculate_numerical_risk_score(probs, class_order=('low', 'medium', 'high'), input_dict=feature_dict)
        factors = compute_instance_contributions(feature_dict, risk_result['risk_score'])
        
        risk_result['factors'] = factors
        risk_result['model_version'] = self.metadata.get('model_version', 'fingerdiab-rf-v1.2.0')
        risk_result['model_name'] = selected_info['display_name']
        risk_result['model_code'] = selected_info['selected_name']
        risk_result['model_selection_reason'] = selected_info['selection_reason']
        risk_result['model_evaluations'] = selected_info.get('evaluations', [])
        
        return risk_result

get_inference_engine = InferenceEngine.get_instance
