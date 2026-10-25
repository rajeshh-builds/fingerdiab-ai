import pandas as pd
import numpy as np
import os
import json
import time
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

def train_and_evaluate_all():
    os.makedirs('model', exist_ok=True)

    # Load dataset
    df = pd.read_csv('fingerprint_diabetes_dataset.csv')
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    print(f"Loaded dataset with {len(df)} records.")

    # Encode categorical features
    le_fingerprint = LabelEncoder()
    df['Fingerprint_Type_Enc'] = le_fingerprint.fit_transform(df['Fingerprint_Type'].str.lower())
    joblib.dump(le_fingerprint, 'model/le_fingerprint.pkl')

    le_family_history = LabelEncoder()
    df['Family_History_Enc'] = le_family_history.fit_transform(df['Family_History'].str.lower())
    joblib.dump(le_family_history, 'model/le_family_history.pkl')

    le_diabetes_risk = LabelEncoder()
    df['Diabetes_Risk_Enc'] = le_diabetes_risk.fit_transform(df['Diabetes_Risk'].str.lower())
    joblib.dump(le_diabetes_risk, 'model/le_diabetes_risk.pkl')

    # Features and target
    feature_cols = ['Fingerprint_Type_Enc', 'Ridge_Count', 'Age', 'BMI', 'Family_History_Enc']
    X = df[feature_cols]
    y = df['Diabetes_Risk_Enc']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_config = {
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    metrics_report = {}
    classes = list(le_diabetes_risk.classes_) # ['high', 'low', 'medium'] or similar

    for name, clf in models_config.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = round(time.time() - t0, 4)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        cv_scores = cross_val_score(clf, X, y, cv=5)

        cm = confusion_matrix(y_test, y_pred).tolist()

        # Save individual model
        filename = f"model/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
        joblib.dump(clf, filename)

        # Primary model is Random Forest
        if name == "Random Forest":
            joblib.dump(clf, 'model/diabetes_model.pkl')
            # Feature importances
            feature_imp = {feat: round(float(imp), 4) for feat, imp in zip(['Fingerprint Type', 'Ridge Count', 'Age', 'BMI', 'Family History'], clf.feature_importances_)}
        else:
            feature_imp = None

        metrics_report[name] = {
            "name": name,
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "cv_mean_accuracy": round(float(cv_scores.mean()) * 100, 2),
            "cv_std": round(float(cv_scores.std()) * 100, 2),
            "train_time_sec": train_time,
            "confusion_matrix": cm,
            "classes": classes,
            "feature_importance": feature_imp
        }

        print(f"[{name}] Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Train Time: {train_time}s")

    # Write summary metrics JSON
    with open('model/model_metrics.json', 'w') as f:
        json.dump(metrics_report, f, indent=4)

    print("\nAll models trained and exported to model/ directory successfully.")

if __name__ == '__main__':
    train_and_evaluate_all()
