FROM python:3.14.5-slim

RUN pip install -U pip
WORKDIR /app


COPY [ "requirements/requirements-dev.txt", "./" ]
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY [ "deployment/predict.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]