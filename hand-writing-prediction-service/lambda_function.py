import tflite_runtime.interpreter as tflite
import numpy as np
import boto3
import tempfile

# Initialize S3 client
s3_client = boto3.client('s3')

interpreter = tflite.Interpreter(
    model_path="model_hw_newhandpd_aug-ilum_rasnet50.tflite"
)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def predict(X):
    interpreter.set_tensor(input_index, np.float32(X))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return {
        'hw-result': preds[0][1].tolist(),
        'hw-error': None
    }

def lambda_handler(event, context):
    try:
        # Get S3 reference from event
        s3_bucket = event['s3_bucket']
        s3_key = event['s3_key']
        
        # Download numpy array from S3
        with tempfile.NamedTemporaryFile() as tmp:
            s3_client.download_fileobj(s3_bucket, s3_key, tmp)
            tmp.seek(0)
            preprocessed_data = np.load(tmp)
            
        # Make prediction
        return predict(preprocessed_data)
    
    except Exception as e:
        return {
            'hw-result': None,
            'hw-error': str(e)
        }