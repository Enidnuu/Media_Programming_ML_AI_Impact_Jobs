import joblib
import pandas as pd
import numpy as np
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1) Datensatz laden
path = kagglehub.dataset_download("khushikyad001/ai-impact-on-jobs-2030")
csv_path = f"{path}/AI_Impact_on_Jobs_2030.csv"

df = pd.read_csv(csv_path)

# Nur Zeilen mit sinnvollen Daten behalten (optional, zur Sicherheit)
df = df.dropna(subset=["Job_Title", "Education_Level", "Years_Experience", "Average_Salary", "Risk_Category"])

# 2) Zielvariable: Risk_Category (z. B. Low / Medium / High)
label_col = "Risk_Category"
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[label_col])

# 3) Features: nur Dinge, die ein normaler Arbeitnehmer kennt
cat_cols = ["Job_Title", "Education_Level"]
num_cols = ["Years_Experience", "Average_Salary"]

X = df[cat_cols + num_cols]

# 4) Preprocessing: One-Hot-Encoding für Kategorien, Zahlen unverändert
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# 5) Pipeline: Preprocessing + Modell
clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=5000))
    ]
)

# 6) Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7) Trainieren
clf.fit(X_train, y_train)

# 8) Qualität prüfen
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.3f}")

# 9) Speichern
joblib.dump(clf, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("model.pkl und label_encoder.pkl gespeichert.")

