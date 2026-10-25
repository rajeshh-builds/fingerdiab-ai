import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("fingerprint_diabetes_dataset.csv")

# Drop ID
df = df.drop("ID", axis=1)

# Encode categorical variables
le_fp = LabelEncoder()
df["Fingerprint_Type"] = le_fp.fit_transform(df["Fingerprint_Type"])

le_fh = LabelEncoder()
df["Family_History"] = le_fh.fit_transform(df["Family_History"])

le_risk = LabelEncoder()
df["Diabetes_Risk"] = le_risk.fit_transform(df["Diabetes_Risk"])

# Features and target
X = df[["Fingerprint_Type", "Ridge_Count", "Age", "BMI", "Family_History"]]
y = df["Diabetes_Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

print("\nAlgorithm Comparison Results\n")

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {end-start:.4f} seconds")
    print("----------------------------")