import pickle
import numpy as np

input_file = 'model_vm_mdvr-kcl_knn.bin'
with open(input_file, 'rb') as file:
    model = pickle.load(file)
print(f'Model loaded from {input_file}')

def predict(preprocessed_data):
    probabilities = model.predict_proba(preprocessed_data)
    total_probabilities = probabilities.sum(axis=0)
    final_prediction = np.argmax(total_probabilities)
    
    return {
        'vm-result': bool(final_prediction),
        'vm-error': None
    }

def lambda_handler(event, context):
    try:
        if 'data' in event:
            preprocessed_data = np.array(event['data'])
            return predict(preprocessed_data)
        else:
            return {'error': 'No data provided'}
    
    except Exception as e:
        return {
            'vm-result': None,
            'vm-error': str(e)
        }
