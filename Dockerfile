FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_files/model_mi=0.12_depth=None_min_samples_leaf=5.bin", "./"]

EXPOSE 9696

ENV MODEL_FILE="model_mi=0.12_depth=None_min_samples_leaf=5.bin"
ENV PORT=9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]