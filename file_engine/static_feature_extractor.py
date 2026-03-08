import os
import math
from collections import Counter

def calculate_entropy(data):
    if not data:
        return 0
    counter = Counter(data)
    entropy = 0
    for count in counter.values():
        p = count / len(data)
        entropy -= p * math.log2(p)
    return entropy

def count_suspicious_strings(content):
    suspicious_patterns = [
        b"powershell",
        b"cmd.exe",
        b"wget",
        b"curl",
        b"base64",
        b"eval(",
        b"CreateRemoteProcess",
        b"VirtualAlloc",
        b"socket",
        b"http"
    ]
    count = 0
    for pattern in suspicious_patterns:
        if pattern in content:
            count += 1
    return count

def extract_file_features(file_path):
    features = {}

    # File size
    file_size = os.path.getsize(file_path)
    features["file_size"] = file_size

    # Extension risk
    ext = os.path.splitext(file_path)[1].lower()
    risky_extensions = [".exe", ".dll", ".bat", ".js", ".vbs", ".scr"]
    features["is_executable"] = 1 if ext in risky_extensions else 0

    # Read binary content
    with open(file_path, "rb") as f:
        content = f.read()

    # Entropy
    entropy = calculate_entropy(content)
    features["entropy"] = entropy

    # Unique byte count
    byte_counts = Counter(content)
    features["unique_bytes"] = len(byte_counts)

    # Suspicious strings
    features["suspicious_string_count"] = count_suspicious_strings(content)

    # ASCII ratio
    ascii_count = sum(1 for b in content if 32 <= b <= 126)
    features["ascii_ratio"] = ascii_count / len(content) if len(content) > 0 else 0

    return features