FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "trained_models/decision_tree_HostRangeClassifier_v1.bin", "./"]

EXPOSE 9696

ENV MODEL_FILE="decision_tree_HostRangeClassifier_v1.bin"
ENV PORT=9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]