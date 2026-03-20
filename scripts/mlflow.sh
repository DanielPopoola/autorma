#!/usr/bin/env bash

ABS_PATH=$(pwd)

mlflow server \
  --backend-store-uri sqlite:///$ABS_PATH/mlflow_data/mlflow.db \
  --artifacts-destination file:///$ABS_PATH/mlflow_data/artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
