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
import uuid

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

S3_BUCKET = 'parked-dev-lambda-bucket'  

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

def send_to_model(data, function_name):
    """Send data to model using S3 or direct payload for VM model"""

    if function_name == 'parked-dev-vm-model':
        payload = {
            'data': data.tolist()  
        }
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        return json.loads(response['Payload'].read())
    
    file_key = f"temp/{uuid.uuid4()}.npy"
    try:
        with tempfile.NamedTemporaryFile() as tmp:
            np.save(tmp, data)
            tmp.seek(0)
            s3_client.upload_fileobj(tmp, S3_BUCKET, file_key)
        
        payload = {
            's3_bucket': S3_BUCKET,
            's3_key': file_key
        }
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        
        s3_client.delete_object(Bucket=S3_BUCKET, Key=file_key)
        
        return result
        
    except Exception as e:
        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=file_key)
        except:
            pass
        raise e

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
    if 'ni-result' in result:
        prediction_value = result['ni-result']
        prediction_bool = prediction_value >= threshold if prediction_value is not None else None
        processed_result.update({
            'ni-result': prediction_bool,
            'ni-error': result.get('ni-error')
        })
    return processed_result


def lambda_handler(event, context):
    try:
        final_result = {}
        
        if 'vm-url' in event:  
            try:
                file_url = event['vm-url']
                file_path = download_file(file_url)
                preprocessed_data = preprocess_audio(file_path)
                vm_result = send_to_model(preprocessed_data, 'parked-dev-vm-model')
                final_result = vm_result
            except Exception as e:
                final_result.update({
                    'vm-result': None,
                    'vm-error': f"VM processing error: {str(e)}"
                })
            finally:
                # Cleanup VM file
                if 'file_path' in locals() and file_path and os.path.exists(file_path):
                    os.remove(file_path)
        
        if 'hw-url' in event:  
            try:
                file_url = event['hw-url']
                preprocessed_data = preprocess_image_hw(file_url)
                hw_result = send_to_model(preprocessed_data, 'parked-dev-hw-model')
                final_result.update(post_process_result(hw_result, threshold=0.8))
            except Exception as e:
                final_result.update({
                    'hw-result': None,
                    'hw-error': f"HW processing error: {str(e)}"
                })

        if 'ni-url' in event:  
            try:
                file_url = event['ni-url']
                preprocessed_data = preprocess_image_ni(file_url)
                ni_result = send_to_model(preprocessed_data, 'parked-dev-ni-model')
                final_result.update(post_process_result(ni_result, threshold=0.5))
                final_result.update(post_process_result(ni_result))
            except Exception as e:
                final_result.update({
                    'ni-result': None,
                    'ni-error': f"NI processing error: {str(e)}"
                })
        
        if not final_result:
            final_result = {'error': 'No valid URL provided'}
        
        return final_result
    
    except Exception as e:
        return {
            'error': f"Unexpected error: {str(e)}"
        }