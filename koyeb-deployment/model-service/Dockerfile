FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi[standard] \
    prometheus-client \
    pillow \
    timm \
    && pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app.py .
COPY best_model.pth .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]