FROM public.ecr.aws/lambda/python:3.11

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/
COPY lambda_function.py .

COPY ["model_vm_mdvr-kcl_knn.bin", "./"]

CMD ["lambda_function.lambda_handler"]

