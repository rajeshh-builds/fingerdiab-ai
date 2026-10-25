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
from app.services.risk_service import calculate_numerical_risk_score
from app.ml.explainability import compute_instance_contributions

# Canonical artifact paths
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
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance
        
    def load_artifacts(self):
        """Loads model and preprocessing artifacts into memory."""
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
            
    def predict_risk(self, feature_dict, model_choice=None):
        """
        Executes end-to-end inference on validated patient inputs.
        
        Args:
            feature_dict: dict containing age, bmi, family_history, fingerprint_type, ridge_count
            model_choice: optional string to select specific candidate model
            
        Returns:
            dict containing risk_score, model_probability, interpretation,
                 class_probabilities, factors, and model_version.
        """
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("ML model is unavailable or corrupted.")
                
        # Format input DataFrame
        input_row = pd.DataFrame([{
            'Fingerprint_Type': str(feature_dict.get('fingerprint_type', 'loop')).lower(),
            'Ridge_Count': float(feature_dict.get('ridge_count', 18)),
            'Age': float(feature_dict.get('age', 40)),
            'BMI': float(feature_dict.get('bmi', 25.0)),
            'Family_History': 'yes' if str(feature_dict.get('family_history', '')).lower() in {'yes', '1', 'true'} else 'no'
        }])
        
        # Select model if user specified alternate algorithm
        active_model = self.model
        if model_choice and model_choice.lower() != 'default':
            cand_path = os.path.join(MODEL_DIR, f"{model_choice.lower()}_model.pkl")
            if os.path.exists(cand_path):
                try:
                    active_model = joblib.load(cand_path)
                except Exception:
                    active_model = self.model
                    
        # Apply preprocessing
        if self.pipeline is not None:
            X_trans = self.pipeline.transform(input_row)
        else:
            # Fallback direct transform if pipeline missing
            from app.ml.preprocessing import build_preprocessing_pipeline
            pipe = build_preprocessing_pipeline()
            X_trans = pipe.fit_transform(input_row)
            
        # Predict class probabilities
        if hasattr(active_model, 'predict_proba'):
            probs = active_model.predict_proba(X_trans)[0]
        elif hasattr(active_model, 'decision_function'):
            # Calibrate SVM decision function via softmax
            df_vals = active_model.decision_function(X_trans)[0]
            exp_vals = np.exp(df_vals - np.max(df_vals))
            probs = exp_vals / np.sum(exp_vals)
        else:
            pred_idx = active_model.predict(X_trans)[0]
            probs = [0.0, 0.0, 0.0]
            probs[int(pred_idx)] = 1.0
            
        # Derive continuous risk score
        risk_result = calculate_numerical_risk_score(probs, class_order=('low', 'medium', 'high'), input_dict=feature_dict)
        
        # Calculate explainability feature contributions
        factors = compute_instance_contributions(feature_dict, risk_result['risk_score'])
        
        risk_result['factors'] = factors
        risk_result['model_version'] = self.metadata.get('model_version', 'fingerdiab-rf-v1.0.0')
        risk_result['model_name'] = self.metadata.get('model_name', 'Random Forest Classifier')
        
        return risk_result
