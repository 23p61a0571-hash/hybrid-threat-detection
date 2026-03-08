from flask import Flask, render_template, request
from url_engine.hybrid_engine import hybrid_url_prediction
from werkzeug.utils import secure_filename
from url_engine.url_feature_builder import build_feature_vector
from file_engine.static_feature_extractor import extract_file_features
import matplotlib
matplotlib.use('Agg')  # Important for Flask
import matplotlib.pyplot as plt
import os
import os
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# Configuration
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load static file ML model
with open("models/url_feature_columns.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("models/dynamic_url_model_lexical.pkl", "rb") as f:
    url_model = pickle.load(f)
with open("models/static_file_ml_model.pkl", "rb") as f:
    file_model = pickle.load(f)



# ==============================
# Home Page
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# URL Checker
# ==============================
@app.route("/url-check", methods=["GET", "POST"])
def url_check():

    if request.method == "POST":

        user_url = request.form.get("url")

        # Input validation
        if not user_url or user_url.strip() == "":
            return render_template(
                "url_check.html",
                result="Invalid",
                confidence=0
            )

        try:
            # Run hybrid prediction
            features = build_feature_vector(user_url)
            print("Extracted Features:", features)
            prediction = hybrid_url_prediction(user_url)

            print("Prediction Output:", prediction)

            # Build dynamic feature chart
            feature_chart = get_url_feature_chart(user_url)

            return render_template(
                "url_check.html",
                result=prediction["risk_level"],
                confidence=prediction["probability"],
                ml_score=prediction["ml_score"],
                html_score=prediction["html_score"],
                rule_score=prediction["rule_score"],
                feature_labels=feature_chart["labels"],
                feature_values=feature_chart["values"]
            )

        except Exception as e:
            print("URL Check Error:", e)

            return render_template(
                "url_check.html",
                result="Invalid",
                confidence=0
            )

    return render_template("url_check.html")

# ==============================
# File Checker (FULLY WORKING)
# ==============================
@app.route("/file-check", methods=["GET","POST"])
def file_check():

    if request.method == "POST":

        file = request.files.get("file")

        if not file:
            return render_template("file_check.html")

        filepath = os.path.join("uploads", secure_filename(file.filename))
        file.save(filepath)

        features = extract_file_features(filepath)

        feature_values = list(features.values())
        feature_labels = list(features.keys())
        

        prediction = file_model.predict([feature_values])[0]
        prob = file_model.predict_proba([feature_values])[0][1] * 100

        result = "Malicious File ❌" if prediction == 1 else "Safe File ✔"

        return render_template(
            "file_check.html",
            result=result,
            confidence=prob,
            ml_score=round(prob,2),
            rule_score=round(100-prob,2),
            file_details=features,
            feature_labels=feature_labels,
            feature_values=feature_values
        )

    return render_template("file_check.html")
def generate_feature_importance():
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        importances = url_model.feature_importances_

        sorted_indices = np.argsort(importances)[::-1]
        top_n = 10

        top_importances = importances[sorted_indices][:top_n]
        top_features = [feature_names[i] for i in sorted_indices][:top_n]

        colors = plt.cm.plasma(
            [i / top_n for i in range(top_n)]
        )

        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importances[::-1], color=colors)

        plt.xlabel("Importance Score", fontsize=12)
        plt.title("Top 10 URL Feature Importance", fontsize=14, fontweight='bold')

        for i, v in enumerate(top_importances[::-1]):
            plt.text(v + 0.001, i, f"{v:.3f}", va='center')

        os.makedirs("static", exist_ok=True)
        plt.tight_layout()
        plt.savefig("static/feature_importance.png", dpi=300)
        plt.close()

    except Exception as e:
        print("Feature Importance Error:", e)
def get_feature_importance_data():
    import numpy as np

    importances = url_model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    top_n = 10

    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_values = importances[sorted_idx[:top_n]]

    return {
        "labels": top_features,
        "values": top_values.tolist()
    }
def get_url_feature_chart(url):
    features = build_feature_vector(url)

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
        "Suspicious Words",
        "Path Length",
        "Digit Ratio",
        "Special Ratio"
    ]

    return {
        "labels": feature_names,
        "values": features[:len(feature_names)]
    }

# ==============================
# Run Server
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)