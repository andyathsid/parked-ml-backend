FROM python:3.11.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/

COPY ["predict.py", "model_vm_mdvr-kcl_knn.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
