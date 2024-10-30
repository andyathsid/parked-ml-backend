import os
import pickle
import numpy as np
import requests
import tempfile
from urllib.parse import urlparse

input_file = 'model_vm_mdvr-kcl_knn.bin'
with open(input_file, 'rb') as file:
    model = pickle.load(file)
print(f'Model loaded from {input_file}')

def get_filename_from_url(url):
    """Extract filename from URL"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)
    return filename if filename else 'downloaded_file'

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        filename = get_filename_from_url(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def predict(file_url):
    try:
        file_path = download_file(file_url)
        
        processed_data = process_single_file_for_prediction(file_path)
        
        if processed_data is not None:
            X_test = processed_data.values
            scaler = MinMaxScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            
            probabilities = model.predict_proba(X_test_scaled)
            total_probabilities = probabilities.sum(axis=0)
            final_prediction = np.argmax(total_probabilities)
            
            result = {
                'vm-result': bool(final_prediction),
                'vm-error': None
            }
        else:
            result = {
                'vm-result': None,
                'vm-error': 'Failed to process the file'
            }
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return result
    except Exception as e:
        return {
            'vm-result': None,
            'vm-error': str(e)
        }

def lambda_handler(event, context):
    try:
        if 'vm-url' in event:
            file_url = event['vm-url']
            return predict(file_url)
        else:
            return {
                'vm-result': None,
                'vm-error': 'No URL provided'
            }
    
    except Exception as e:
        return {
            'vm-result': None,
            'vm-error': str(e)
        }