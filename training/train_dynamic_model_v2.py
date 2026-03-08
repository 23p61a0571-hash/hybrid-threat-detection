import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from url_engine.url_feature_builder import build_feature_vector

print("Loading raw URL dataset...")
data = pd.read_csv("data/url_dataset.csv", encoding="latin1")

# Rename columns properly
data.columns = ["URL", "Label"]

# Convert labels
data["Label"] = data["Label"].map({
    "bad": 0,
    "good": 1
})

print("Extracting lexical features...")
X = []
y = []

for _, row in data.iterrows():
    features = build_feature_vector(str(row["URL"]))
    X.append(features)
    y.append(row["Label"])

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# Train-test split (Stratified recommended)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
ratio = sum(y_train == 0) / sum(y_train == 1)

print("Training model...")
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=ratio,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save model
with open("models/dynamic_url_model_lexical.pkl", "wb") as f:
    pickle.dump(model, f)

print("Dynamic URL model saved successfully.")
