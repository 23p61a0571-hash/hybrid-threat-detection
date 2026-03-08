import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading malware dataset...")
data = pd.read_csv("data/file_dataset.csv")

# -----------------------------
# 1️⃣ Data Cleaning
# -----------------------------
print("Cleaning dataset...")

# Drop non-numeric identifier column
if "hash" in data.columns:
    data = data.drop("hash", axis=1)

# Convert classification to numeric
data["classification"] = data["classification"].map({
    "benign": 0,
    "malware": 1
})

# Remove any possible missing values
data = data.dropna()

# -----------------------------
# 2️⃣ Split Features & Labels
# -----------------------------
X = data.drop("classification", axis=1)
y = data["classification"]

print("Dataset shape:", X.shape)

# -----------------------------
# 3️⃣ Stratified Train/Test Split
# -----------------------------
print("Splitting dataset (stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4️⃣ Train Model
# -----------------------------
print("Training RandomForest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Cross-Validation (Realistic Evaluation)
# -----------------------------
print("Performing 5-fold cross validation...")

cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)

print("Cross-validation accuracy:", cv_scores.mean())
print("CV Std Dev:", cv_scores.std())

# -----------------------------
# 6️⃣ Test Set Evaluation
# -----------------------------
print("\nEvaluating on test set...")

preds = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
with open("models/file_ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nFile ML model saved successfully.")