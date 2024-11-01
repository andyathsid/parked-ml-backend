import os
import requests
import tempfile
import numpy as np
import boto3
from urllib.parse import urlparse
from scripts.mdvr_extraction import process_single_file_for_prediction
from sklearn.preprocessing import MinMaxScaler
import json
from keras_image_helper import create_preprocessor

lambda_client = boto3.client('lambda')

def get_filename_from_url(url):
    """Extract filename from URL"""
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path) or 'downloaded_file'

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        filename = get_filename_from_url(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def preprocess_audio(file_path):
    """Preprocess audio by splitting, extracting features, and scaling"""
    processed_data = process_single_file_for_prediction(file_path)
    if processed_data is not None:
        X_test = processed_data.values
        scaler = MinMaxScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        return X_test_scaled
    else:
        raise Exception('Failed to process the audio file')

def preprocess_image_hw(url):
    """Resize image and prepare as numpy array for model input"""
    preprocessor = create_preprocessor("resnet50", target_size=(224, 224))
    X = preprocessor.from_url(url)
    return X.astype(np.float32)

def preprocess_image_ni(url):
    """Resize image and prepare as numpy array for VGG16 neuroimaging model"""
    preprocessor = create_preprocessor("vgg16", target_size=(512, 512))
    X = preprocessor.from_url(url)
    return X.astype(np.float32)

def send_to_model(preprocessed_data, model_name):
    """Send preprocessed data to the model Lambda function and get response"""
    response = lambda_client.invoke(
        FunctionName=model_name,
        InvocationType='RequestResponse',
        Payload=json.dumps({'data': preprocessed_data.tolist()})
    )
    result = json.loads(response['Payload'].read())
    return result

def post_process_result(result, threshold=0.5):
    """Post-process the model result using a threshold"""
    processed_result = {}
    
    if 'hw-result' in result:
        prediction_value = result['hw-result']
        prediction_bool = prediction_value >= threshold if prediction_value is not None else None
        processed_result.update({
            'hw-result': prediction_bool,
            'hw-error': result.get('hw-error')
        })
    if 'vm-result' in result:
        prediction_value = result['vm-result']
        prediction_bool = prediction_value >= threshold if prediction_value is not None else None
        processed_result.update({
            'vm-result': prediction_bool,
            'vm-error': result.get('vm-error')
        })
    return processed_result

def lambda_handler(event, context):
    try:
        file_path = None
        final_result = {}
        
        if 'vm-url' in event:  
            file_url = event['vm-url']
            file_path = download_file(file_url)
            preprocessed_data = preprocess_audio(file_path)
            vm_result = send_to_model(preprocessed_data, 'parked-dev-vm-model')
            final_result.update(post_process_result(vm_result))
            
            # Cleanup VM file
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        
        if 'hw-url' in event:  
            file_url = event['hw-url']
            preprocessed_data = preprocess_image_hw(file_url)
            preprocessed_data.astype(np.float32)
            hw_result = send_to_model(preprocessed_data, 'parked-dev-hw-model')
            final_result.update(post_process_result(hw_result))

        if 'ni-url' in event:  
            file_url = event['ni-url']
            preprocessed_data = preprocess_image_ni(file_url)
            preprocessed_data.astype(np.float32)
            ni_result = send_to_model(preprocessed_data, 'parked-dev-ni-model')
            final_result.update(post_process_result(ni_result))
        
        if not final_result:
            final_result = {'error': 'No valid URL provided'}
        
        return final_result
    
    except Exception as e:
        return {
            'vm-result': None,
            'vm-error': str(e),
            'hw-result': None,
            'hw-error': str(e),
            'ni-result': None,
            'ni-error': str(e)
        }