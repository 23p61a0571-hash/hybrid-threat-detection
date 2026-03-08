import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from url_engine.url_feature_builder import build_feature_vector

print("Loading URL dataset...")
data = pd.read_csv("data/url_dataset.csv")

print("Extracting lexical features...")

X = []
y = []

for _, row in data.iterrows():
    url = str(row["URL"])
    label = row["label"]

    features = build_feature_vector(url)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# ✅ Manually define feature names (same order as builder)
feature_names = [
    "URL Length",
    "Hostname Length",
    "@ Symbol",
    "Hyphen Count",
    "Dot Count",
    "Digit Count",
    "Special Char Count",
    "HTTPS",
    "Subdomain Count",
    "Suspicious Word Count",
    "Path Length",
    "Digit Ratio",
    "Special Char Ratio"
]

# Pad to 21 if needed
while len(feature_names) < X.shape[1]:
    feature_names.append(f"Extra Feature {len(feature_names)+1}")

# Save feature names
with open("models/url_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc"
)

model.fit(X_train, y_train)

print("\nEvaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("models/dynamic_url_model_lexical.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model and feature names saved successfully.")