import json
import requests
import tempfile
import os
from scripts.mdvr_extraction import process_single_file_for_prediction  
from urllib.parse import urlparse
import boto3

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.basename(urlparse(url).path) or "downloaded_file"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def preprocess_and_route(file_url, model_name):
    file_path = download_file(file_url)
    processed_data = None
    
    if model_name == 'model_vm_mdvr':
        processed_data = process_single_file_for_prediction(file_path)
    elif model_name == 'another_model':
        pass
    
    if processed_data is None:
        raise Exception("Preprocessing failed.")
    
    os.remove(file_path)
    
    data_payload = processed_data.to_dict()  
    
    response = lambda_client.invoke(
        FunctionName=f"{model_name}_lambda",
        InvocationType='RequestResponse',
        Payload=json.dumps(data_payload)
    )
    result = json.load(response['Payload'])
    
    return result

def lambda_handler(event, context):
    try:
        if 'file-url' in event and 'model-name' in event:
            file_url = event['file-url']
            model_name = event['model-name']
            result = preprocess_and_route(file_url, model_name)
            return result
        else:
            return {'error': 'Missing file URL or model name'}
    except Exception as e:
        return {'error': str(e)}
