import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "/tmp/model_vm_mdvr-kcl_knn.bin"
SCALER_PATH = "/tmp/scaler_vm_mdvr.pkl"

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

def predict(data):
    X_test = np.array(data)
    X_test_scaled = scaler.transform(X_test)
    
    probabilities = model.predict_proba(X_test_scaled)
    total_probabilities = probabilities.sum(axis=0)
    final_prediction = int(np.argmax(total_probabilities))
    
    return {'vm-result': bool(final_prediction), 'vm-error': None}

def lambda_handler(event, context):
    try:
        if 'processed_data' in event:
            return predict(event['processed_data'])
        else:
            return {'vm-result': None, 'vm-error': 'Processed data not provided'}
    except Exception as e:
        return {'vm-result': None, 'vm-error': str(e)}
