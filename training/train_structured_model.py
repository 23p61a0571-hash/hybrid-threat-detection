import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from url_engine.url_feature_builder import build_feature_vector

print("Loading URL dataset...")

# Load dataset
data = pd.read_csv("data/url_dataset.csv")

# Make sure correct column names
if "label" not in data.columns:
    raise ValueError("Dataset must contain 'label' column")

if "URL" not in data.columns:
    raise ValueError("Dataset must contain 'URL' column")

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

# Train-test split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance
ratio = sum(y_train == 0) / sum(y_train == 1)

print("Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False
)

model.fit(X_train, y_train)

print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save model
with open("models/dynamic_url_model_lexical.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as dynamic_url_model_lexical.pkl")
print("Training complete.")
