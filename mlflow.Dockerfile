FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir mlflow

EXPOSE 5000

ENV MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING=TRUE

# Note: the artifact root uses a local path inside container
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow_data/mlflow.db", \
     "--artifacts-destination", "file:///mlflow_data/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
