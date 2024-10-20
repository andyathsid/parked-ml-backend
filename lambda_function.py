import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scripts.mdvr_extraction import process_single_file_for_prediction
import base64

# Load the model when the Lambda function is first initialized
input_file = 'model_vm_mdvr-kcl_knn.bin'
with open(input_file, 'rb') as file:
    model = pickle.load(file)

print(f'Model loaded from {input_file}')

def predict(file_content, filename):
    file_path = f"/tmp/{filename}"
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    processed_data = process_single_file_for_prediction(file_path)
    
    if processed_data is not None:
        X_test = processed_data.values
        scaler = MinMaxScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        probabilities = model.predict_proba(X_test_scaled)
        total_probabilities = probabilities.sum(axis=0)
        final_prediction = np.argmax(total_probabilities)
        
        result = {
            'detection': bool(final_prediction) 
        }
    else:
        result = {'error': 'Failed to process the file.'}
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return result

def test_lambda_function(file_path):
    with open(file_path, 'rb') as f:
        file_content = f.read()

    encoded_file = base64.b64encode(file_content)

    decoded_file = base64.b64decode(encoded_file)

    result = predict(decoded_file, os.path.basename(file_path))

    print("Prediction result:", result)

def lambda_handler(event, context):
    try:
        if 'file' in event and 'filename' in event:
            file_content = base64.b64decode(event['file'])
            filename = event['filename']
        else:
            return {
                'statusCode': 400,
                'body': 'No file or filename provided.'
            }
        
        result = predict(file_content, filename)
        
        return {
            'statusCode': 200,
            'body': result
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }
