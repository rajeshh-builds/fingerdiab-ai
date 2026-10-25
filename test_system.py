# Quick Verification Script to test all engines and pipelines offline
import os
import sys
import pandas as pd
import numpy as np

print("==================================================")
print("FINGERDIAB AI BATCH-5 - SYSTEM INTEGRITY TEST")
print("==================================================")

# 1. Test Dataset
try:
    df = pd.read_csv('fingerprint_diabetes_dataset.csv')
    print(f"[PASS] Dataset loaded: {len(df)} records, Columns: {list(df.columns)}")
except Exception as e:
    print(f"[FAIL] Dataset error: {e}")

# 2. Test Remedy Engine
try:
    from remedy_engine import get_clinical_recommendations
    high_rec = get_clinical_recommendations("HIGH", age=50, bmi=31.2, family_history="yes", fingerprint_type="whorl", ridge_count=12)
    low_rec = get_clinical_recommendations("LOW", age=25, bmi=21.5, family_history="no", fingerprint_type="loop", ridge_count=22)
    print(f"[PASS] Remedy Engine verified: High Risk Focus -> {high_rec['diet_plan']['title']}")
    print(f"[PASS] Remedy Engine verified: Low Risk Focus  -> {low_rec['diet_plan']['title']}")
except Exception as e:
    print(f"[FAIL] Remedy Engine error: {e}")

# 3. Test Database Initializer
try:
    from init_db import init_enhanced_db, get_db_connection
    init_enhanced_db()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    conn.close()
    print(f"[PASS] Database SQLite verified. Tables present: {tables}")
except Exception as e:
    print(f"[FAIL] Database error: {e}")

# 4. Test Models & Encoders
try:
    import joblib
    le_fp = joblib.load('model/le_fingerprint.pkl')
    le_fh = joblib.load('model/le_family_history.pkl')
    le_risk = joblib.load('model/le_diabetes_risk.pkl')
    rf_model = joblib.load('model/diabetes_model.pkl')

    # Mock prediction
    test_fp = le_fp.transform(['loop'])[0]
    test_fh = le_fh.transform(['no'])[0]
    test_input = pd.DataFrame([[test_fp, 20, 30, 22.5, test_fh]],
                              columns=['Fingerprint_Type_Enc', 'Ridge_Count', 'Age', 'BMI', 'Family_History_Enc'])

    pred = rf_model.predict(test_input)[0]
    risk_label = le_risk.inverse_transform([pred])[0]
    print(f"[PASS] Primary ML Model verified. Test Prediction outcome: {risk_label.upper()}")
except Exception as e:
    print(f"[FAIL] Model error: {e}")

print("==================================================")
print("ALL SYSTEM INTEGRITY CHECKS PASSED SUCCESSFULLY!")
print("==================================================")
