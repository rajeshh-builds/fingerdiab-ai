"""
FingerDiab AI — ML Preprocessing and Feature Pipeline
Implements leak-free sklearn Pipeline & ColumnTransformer for feature encoding & scaling.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Controlled canonical values
FEATURE_COLUMNS = ['Fingerprint_Type', 'Ridge_Count', 'Age', 'BMI', 'Family_History']
NUMERICAL_FEATURES = ['Ridge_Count', 'Age', 'BMI']
CATEGORICAL_FEATURES = ['Fingerprint_Type', 'Family_History']

# Target mapping (Standardized Ordinal)
TARGET_CLASS_MAP = {
    'low': 0,
    'medium': 1,
    'high': 2
}
REVERSE_TARGET_MAP = {
    0: 'low',
    1: 'medium',
    2: 'high'
}

class ClinicalFeatureSanitizer(BaseEstimator, TransformerMixin):
    """
    Sanitizes raw patient and biometric inputs:
    - Normalizes case and trims whitespace
    - Validates numeric bounds
    - Handles missing values gracefully with robust medians
    """
    def __init__(self):
        self.median_values_ = {}
        
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        else:
            X_df = X.copy()
            
        for num_col in NUMERICAL_FEATURES:
            if num_col in X_df.columns:
                self.median_values_[num_col] = float(X_df[num_col].median())
        return self
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        else:
            X_df = X.copy()
            
        # Ensure column existence
        for col in FEATURE_COLUMNS:
            if col not in X_df.columns:
                raise ValueError(f"Missing required feature column: {col}")
                
        # String normalization
        X_df['Fingerprint_Type'] = X_df['Fingerprint_Type'].astype(str).str.lower().str.strip()
        X_df['Family_History'] = X_df['Family_History'].astype(str).str.lower().str.strip()
        
        # Valid category fallbacks
        valid_fp = {'arch', 'loop', 'whorl'}
        X_df['Fingerprint_Type'] = X_df['Fingerprint_Type'].apply(lambda v: v if v in valid_fp else 'loop')
        
        valid_fh = {'yes', 'no', '1', '0', 'true', 'false'}
        X_df['Family_History'] = X_df['Family_History'].apply(lambda v: 'yes' if str(v).lower() in {'yes', '1', 'true'} else 'no')
        
        # Numerical coercion & clipping to biological plausibility
        X_df['Age'] = pd.to_numeric(X_df['Age'], errors='coerce').fillna(self.median_values_.get('Age', 43.0)).clip(18.0, 100.0)
        X_df['BMI'] = pd.to_numeric(X_df['BMI'], errors='coerce').fillna(self.median_values_.get('BMI', 26.5)).clip(15.0, 55.0)
        X_df['Ridge_Count'] = pd.to_numeric(X_df['Ridge_Count'], errors='coerce').fillna(self.median_values_.get('Ridge_Count', 18.0)).clip(0.0, 50.0)
        
        return X_df[FEATURE_COLUMNS]


def build_preprocessing_pipeline():
    """
    Constructs a complete leak-free preprocessing pipeline.
    Combines sanitizer, one-hot encoding for categorical variables,
    and standard scaling for continuous physiological vitals.
    """
    column_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(categories=[['arch', 'loop', 'whorl'], ['no', 'yes']], 
                                  handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('sanitizer', ClinicalFeatureSanitizer()),
        ('preprocessor', column_preprocessor)
    ])
    
    return pipeline
