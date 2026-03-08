import os
import pandas as pd
from file_engine.static_feature_extractor import extract_file_features

DATASET_PATH = "file_dataset"

rows = []

for label_folder in ["benign", "malicious"]:
    folder_path = os.path.join(DATASET_PATH, label_folder)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            features = extract_file_features(file_path)
            features["label"] = 0 if label_folder == "benign" else 1
            rows.append(features)
        except Exception as e:
            print("Error processing:", filename, e)

df = pd.DataFrame(rows)

df.to_csv("data/generated_file_features.csv", index=False)

print("Dataset generated successfully.")
print("Shape:", df.shape)