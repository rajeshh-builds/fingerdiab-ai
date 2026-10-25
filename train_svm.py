import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# load dataset
df = pd.read_csv("fingerprint_diabetes_dataset.csv")

# correct column names
X = df[['Age', 'BMI', 'Ridge_Count', 'Fingerprint_Type', 'Family_History']]
y = df['Diabetes_Risk']

# encoding
X = pd.get_dummies(X)
y = pd.factorize(y)[0]
joblib.dump(X.columns, "model/svm_columns.pkl")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train SVM
svm = SVC()
svm.fit(X_train, y_train)

# save model
joblib.dump(svm, "model/svm_model.pkl")

print("SVM model trained and saved successfully.")