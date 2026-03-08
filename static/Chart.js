import json

def get_feature_importance_data():
    importances = url_model.feature_importances_

    # Sort descending
    sorted_indices = importances.argsort()[::-1]

    top_n = 10
    top_importances = importances[sorted_indices][:top_n]
    top_features = [feature_names[i] for i in sorted_indices][:top_n]

    return {
        "labels": top_features,
        "values": top_importances.tolist()
    }