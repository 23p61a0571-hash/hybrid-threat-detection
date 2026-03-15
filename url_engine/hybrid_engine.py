import pickle
import numpy as np
from url_engine.url_feature_builder import build_feature_vector
from url_engine.html_analyzer import analyze_html

# Load dynamic lexical model
with open("models/dynamic_url_model_lexical.pkl", "rb") as f:
    dynamic_model = pickle.load(f)


def hybrid_url_prediction(url):
    try:
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        # =============================
        # 1️⃣ Lexical ML Prediction
        # =============================
        features = build_feature_vector(url)
        features = np.array(features).reshape(1, -1)

        # IMPORTANT: class 0 = phishing
        ml_proba = float(dynamic_model.predict_proba(features)[0][0])

        # =============================
        # 2️⃣ HTML Risk Analysis
        # =============================
        html_score = analyze_html(url)  # 0–25

        # Normalize HTML score to 0–1
        html_proba = html_score / 25.0

        # =============================
        # 3️⃣ Rule-Based Risk
        # =============================
        rule_score = 0.0
        suspicious_keywords = ["login", "verify", "update", "secure", "bank", "account"]

        for word in suspicious_keywords:
            if word in url.lower():
                rule_score += 0.15

        rule_score = min(rule_score, 1.0)

        # =============================
        # 4️⃣ Trusted TLD Adjustment
        
        # =============================
        trusted_tlds = [".ac.in", ".edu", ".gov", ".org"]
        trusted_domains = ["github.com", "google.com", "microsoft.com", "amazon.com"]
        if any(domain in url.lower() for domain in trusted_domains):
            ml_proba *= 0.2

        if any(tld in url.lower() for tld in trusted_tlds):
            # Reduce malicious probability slightly
            ml_proba *= 0.6
            rule_score *= 0.5

        # =============================
        # 5️⃣ Final Hybrid Score
        # =============================
        final_score = (
            ml_proba * 0.70 +
            html_proba * 0.20 +
            rule_score * 0.10
        )

        final_score = min(max(final_score, 0.0), 1.0)

        # =============================
        # 6️⃣ Risk Classification
        # =============================
        if final_score >= 0.65:
            risk_level = "Malicious"
        elif final_score >= 0.40:
            risk_level = "Suspicious"
        else:
            risk_level = "Safe"

        result = {
            "probability": round(final_score * 100, 2),
            "risk_level": risk_level,
            "ml_score": round(ml_proba * 100, 2),
            "html_score": html_score,
            "rule_score": round(rule_score * 100, 2)
        }

        print("Prediction Output:", result)

        return result

    except Exception as e:
        print("Hybrid Engine Error:", e)
        return {
            "probability": 0,
            "risk_level": "Invalid",
            "ml_score": 0,
            "html_score": 0,
            "rule_score": 0
        }

