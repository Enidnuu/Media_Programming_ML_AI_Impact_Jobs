import joblib
import pandas as pd
import numpy as np
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Datensatz von Kaggle laden
path = kagglehub.dataset_download("khushikyad001/ai-impact-on-jobs-2030")
csv_path = f"{path}/AI_Impact_on_Jobs_2030.csv"

# CSV einlesen
df = pd.read_csv(csv_path)

# Zielvariable auswählen
y = LabelEncoder().fit_transform(df["Risk_Category"])

# Features auswählen (Job_Title raus, Risk_Category raus)
X = df.drop(columns=["Risk_Category", "Job_Title"])

# One-Hot-Encoding für Textfeatures (z. B. Education_Level)
X = pd.get_dummies(X)

# In NumPy-Array konvertieren (wie iris.data)
X = X.values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression wie bei Iris
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Modell-accuracy: {accuracy:.3f}")

# Speichern
joblib.dump(model, "model.pkl")
joblib.dump(X_train, "feature_names.pkl")

print("Fertig!")
