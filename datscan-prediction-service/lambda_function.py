import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(
    model_path="model_hw_newhandpd_aug-ilum_vgg16.tflite"  
)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


def predict(X):
    if X.shape != (1, 512, 512, 3):
        return {
            'hw-result': None,
            'hw-error': f'Invalid input shape. Expected (1, 512, 512, 3), got {X.shape}'
        }
    
    interpreter.set_tensor(input_index, np.float32(X))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return {
        'hw-result': preds[0][1].tolist(),
        'hw-error': None
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
            'hw-result': None,
            'hw-error': str(e)
        }
