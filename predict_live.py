import joblib
import pandas as pd
import json
from utils.constants import LABEL_MAP

# Load the model into memory
model = joblib.load('models/ddos_rf_model.pkl')

# Load feature list from the JSON file in models
with open('models/ddos_rf_features.json') as f:
    FEATURES = json.load(f)

#predicts here using flow data dict data with the features and predicts using the data given in the parameter
def predict_from_dict(flow_data_dict):
    df = pd.DataFrame([flow_data_dict])
    df = df.reindex(columns=FEATURES, fill_value=0)
    prediction = model.predict(df)[0]
    print(f"Result: {LABEL_MAP[prediction]}")
    return prediction

# Example test
if __name__ == "__main__":
    sample = {
        'Flow Duration': 40335006,
        'Total Fwd Packets': 9,
        'Total Backward Packets': 10,
        'Total Length of Fwd Packets': 200,
        'Total Length of Bwd Packets': 0
    }
    predict_from_dict(sample)
