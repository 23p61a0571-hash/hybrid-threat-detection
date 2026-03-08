from feature_extractor import extract_url_features

url = "https://secure-login-bank-update.com/login.php?id=123"
features = extract_url_features(url)

print(features)