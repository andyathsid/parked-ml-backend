FROM public.ecr.aws/lambda/python:3.11

RUN pip install keras-image-helper
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY scripts/ ./scripts/
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]

