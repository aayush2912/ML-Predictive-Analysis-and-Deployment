# Base Image
FROM python:3.10

RUN pip install pipenv

WORKDIR /app

# Copy Dependencies
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

# Port Configuration
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]