import os
import random

OUTPUT_DIR = "file_dataset/malicious"
os.makedirs(OUTPUT_DIR, exist_ok=True)

suspicious_strings = [
    b"powershell -nop -enc",
    b"cmd.exe /c",
    b"wget http://malicious.com",
    b"curl http://evil.com",
    b"eval(base64_decode)",
    b"CreateRemoteProcess",
    b"VirtualAlloc",
    b"socket connect",
]

def generate_high_entropy_bytes(size=5000):
    return os.urandom(size)

for i in range(15):
    filename = f"malware_sample_{i}.bin"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "wb") as f:
        # Write random high entropy bytes
        f.write(generate_high_entropy_bytes())

        # Inject suspicious strings
        for s in suspicious_strings:
            f.write(b"\n" + s + b"\n")

print("Simulated malicious samples created.")