FROM python:3.12-slim

WORKDIR /app


COPY orchestrator/batch_inference.py /app/
COPY orchestrator/config.py /app/
COPY orchestrator/metrics_pusher.py /app/


RUN pip install --no-cache-dir requests prometheus-client pydantic-settings

# Note: data/ directory will be mounted as volume, not copied
# This allows adding new images without rebuilding container

# Command to run batch inference
CMD ["python", "batch_inference.py"]
