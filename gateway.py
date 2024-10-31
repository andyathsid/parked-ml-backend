#!/usr/bin/env python
# coding: utf-8

import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('resnet50', target_size=(224, 224))

def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'model_hw_newhandpd_aug-ilum_resnet50'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_2'].CopyFrom(np_to_protobuf(X))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['dense'].float_val
    return preds[1]


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=30.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    url = 'https://github.com/andyathsid/parked-ml-backend/blob/dev-alt/hand-writing-prediction-service/sp1-P1.jpg?raw=true'
    response = predict(url)
    print(response)
    # app.run(debug=True, host='0.0.0.0', port=9696)