from .url_feature_builder import build_feature_vector

url = "https://example.com"
features = build_feature_vector(url)

print("Feature Vector Length:", len(features))
print("Feature Vector:", features)