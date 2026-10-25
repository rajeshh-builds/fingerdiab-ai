import joblib
import pandas as pd

# load model + columns
svm = joblib.load("model/svm_model.pkl")
cols = joblib.load("model/svm_columns.pkl")

# inputs
age = int(input("Enter age: "))
bmi = float(input("Enter BMI: "))
ridge_count = int(input("Enter ridge count: "))
fingerprint_type = input("Enter fingerprint type (arch/loop/whorl): ")
family_history = input("Family history (yes/no): ")

# create input
sample = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "Ridge_Count": ridge_count,
    "Fingerprint_Type": fingerprint_type,
    "Family_History": family_history
}])

# encoding
sample = pd.get_dummies(sample)

# align columns
sample = sample.reindex(columns=cols, fill_value=0)

# prediction
svm_pred = svm.predict(sample)
labels = ["Low", "Medium", "High"]
print("SVM Prediction:", labels[svm_pred[0]])
