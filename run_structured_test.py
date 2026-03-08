import pandas as pd
from url_engine.url_predictor import predict_structured

# Load one sample from dataset
data = pd.read_csv("data/url_dataset.csv")
sample = data.iloc[0]

features = sample.drop(["url", "status"]).to_dict()

result = predict_structured(features)

print(result)