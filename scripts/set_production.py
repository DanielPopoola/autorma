import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = mlflow.MlflowClient()

# Set alias for version 1
client.set_registered_model_alias(
    name="refund-classifier",
    alias="production",
    version="1"
)

print("✓ Set version 1 to production alias")