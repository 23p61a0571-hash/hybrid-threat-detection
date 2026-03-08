import pickle
import numpy as np
from url_engine.url_feature_builder import build_feature_vector

with open("models/dynamic_url_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_url_live(url):
    try:
        print("Input URL:", url)

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        print("Normalized URL:", url)

        features = build_feature_vector(url)
        print("Feature length:", len(features))

        features = np.array(features).reshape(1, -1)

        print("Model expects:", model.n_features_in_)

        probability = model.predict_proba(features)[0][1]

        print("Probability:", probability)

        if probability < 0.40:
            risk_level = "Safe"
        elif 0.40 <= probability < 0.70:
            risk_level = "Suspicious"
        else:
            risk_level = "Malicious"

        return {
            "prediction": int(probability > 0.60),
            "probability": round(float(probability) * 100, 2),
            "risk_level": risk_level
        }

    except Exception as e:
        print("FULL ERROR:", e)
        return {
            "prediction": 0,
            "probability": 0,
            "risk_level": "Invalid"
        }