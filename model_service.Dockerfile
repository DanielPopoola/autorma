FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy ONLY what model-service needs
COPY model_service/app.py /app/
COPY model_service/config.py /app/
COPY pyproject.toml /app/

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi[standard] mlflow pillow timm prometheus-client \
    && pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu      

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
