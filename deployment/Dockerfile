FROM python:3.14.5-slim

RUN pip install -U pip
WORKDIR /app


COPY [ "requirements/requirements-dev.txt", "./" ]
RUN pip install --no-cache-dir -r requirements-dev.txt

# comment out copying of checkpoint and vocab when pulling info from mlflow
# COPY [ "sms_spam.ckpt", "./" ]
# COPY [ "vocab.pt", "./" ]

COPY [ "deployment/predict.py", "./" ]

COPY [ "spam_checker", "./spam_checker" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]