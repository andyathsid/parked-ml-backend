import tflite_runtime.interpreter as tflite
import numpy as np
import boto3
import tempfile

s3_client = boto3.client('s3')

interpreter = tflite.Interpreter(
    model_path="model_ni_ppmi_vgg16.tflite"  
)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def predict(X):
    if X.shape != (1, 512, 512, 3):
        return {
            'ni-result': None,
            'ni-error': f'Invalid input shape. Expected (1, 512, 512, 3), got {X.shape}'
        }
    
    interpreter.set_tensor(input_index, np.float32(X))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return {
        'ni-result': preds[0][0].tolist(),
        'ni-error': None
    }

def lambda_handler(event, context):
    try:
        s3_bucket = event['s3_bucket']
        s3_key = event['s3_key']
        
        with tempfile.NamedTemporaryFile() as tmp:
            s3_client.download_fileobj(s3_bucket, s3_key, tmp)
            tmp.seek(0)
            preprocessed_data = np.load(tmp)
            
        return predict(preprocessed_data)
    
    except Exception as e:
        return {
            'ni-result': None,
            'ni-error': str(e)
        }