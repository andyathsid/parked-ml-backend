import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


interpreter = tflite.Interpreter(
    model_path="model_hw_newhandpd_aug-ilum_rasnet50.tflite"
)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
preprocessor = create_preprocessor("resnet50", target_size=(224, 224))


def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return {"prediction": preds[0][1].tolist()}


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
