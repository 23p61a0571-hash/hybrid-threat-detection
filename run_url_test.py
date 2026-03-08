from url_engine.url_feature_builder import build_feature_vector
from url_engine.url_predictor import predict_url

url = "https://example.com"

# Step 1: Build features
features = build_feature_vector(url)

print("Feature Vector Length:", len(features))

# Step 2: Predict using trained model
result = predict_url(features)

print("\nPrediction Result:")
print(result)