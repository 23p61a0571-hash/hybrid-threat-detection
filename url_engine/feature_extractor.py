import re
import math
from urllib.parse import urlparse
from collections import Counter

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "bank", "secure",
    "account", "confirm", "password", "signin"
]

def calculate_entropy(url):
    counter = Counter(url)
    length = len(url)
    entropy = 0.0
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy

def has_ip_address(domain):
    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return 1 if re.match(pattern, domain) else 0

def count_suspicious_keywords(url):
    count = 0
    for word in SUSPICIOUS_KEYWORDS:
        if word in url.lower():
            count += 1
    return count

def extract_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {}

    # Basic Length Features
    features["url_length"] = len(url)
    features["domain_length"] = len(domain)
    features["path_length"] = len(path)

    # Character Counts
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["has_at_symbol"] = 1 if "@" in url else 0

    # HTTPS flag
    features["https_flag"] = 1 if parsed.scheme == "https" else 0

    # IP usage
    features["has_ip"] = has_ip_address(domain)

    # Subdomain count
    features["subdomain_count"] = domain.count(".") - 1 if domain.count(".") > 1 else 0

    # Entropy
    features["entropy"] = calculate_entropy(url)

    # Suspicious keyword count
    features["suspicious_keyword_count"] = count_suspicious_keywords(url)

    return features