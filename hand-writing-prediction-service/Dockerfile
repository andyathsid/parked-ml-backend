FROM public.ecr.aws/lambda/python:3.11

RUN pip uninstall numpy
RUN pip install "numpy<2.0"
RUN pip install keras-image-helper
RUN pip install boto3 
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp311-cp311-linux_x86_64.whl?raw=true

COPY model_hw_newhandpd_aug-ilum_rasnet50.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
CMD [ "lambda_function.lambda_handler" ]