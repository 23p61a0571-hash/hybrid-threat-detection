import re
from urllib.parse import urlparse


def build_feature_vector(url):
    try:
        parsed = urlparse(url)

        hostname = parsed.hostname or ""
        path = parsed.path or ""

        # Basic lexical features
        url_length = len(url)
        hostname_length = len(hostname)

        at_symbol = 1 if "@" in url else 0
        hyphen_count = url.count("-")
        dot_count = url.count(".")
        digit_count = sum(c.isdigit() for c in url)
        special_char_count = len(re.findall(r"[^\w]", url))

        https = 1 if parsed.scheme == "https" else 0

        subdomain_count = hostname.count(".") - 1 if hostname.count(".") > 0 else 0

        suspicious_words = ["login", "verify", "update", "secure", "bank", "account"]
        suspicious_word_count = sum(word in url.lower() for word in suspicious_words)

        path_length = len(path)

        digit_ratio = digit_count / url_length if url_length > 0 else 0
        special_ratio = special_char_count / url_length if url_length > 0 else 0

        features = [
            url_length,
            hostname_length,
            at_symbol,
            hyphen_count,
            dot_count,
            digit_count,
            special_char_count,
            https,
            subdomain_count,
            suspicious_word_count,
            path_length,
            digit_ratio,
            special_ratio
        ]

        # Pad to 21 features (to match dynamic model)
        while len(features) < 21:
            features.append(0)

        return features

    except Exception as e:
        print("Feature Builder Error:", e)
        return [0] * 21