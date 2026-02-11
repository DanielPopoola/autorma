import json
from pathlib import Path

import mlflow
import torch

MODEL_PATH = Path("models/v1/best_model.pth")
METADATA_PATH = Path("models/v1/training_metadata.json")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


with open(METADATA_PATH) as f:
    metadata = json.load(f)


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("refund-classifier")

with mlflow.start_run(run_name="efficientnet_b0_v1") as run:
    mlflow.log_params(
        {
            "model_architecture": metadata["model_architecture"],
            "num_classes": metadata["num_classes"],
            "img_size": metadata["img_size"],
            "batch_size": metadata["batch_size"],
            "num_epochs": metadata["num_epochs"],
            "learning_rate": metadata["learning_rate"],
        }
    )

    mlflow.log_metrics(
        {
            "best_val_acc": metadata["best_val_acc"],
            "test_acc": metadata["test_acc"],
            "final_train_acc": metadata["history"]["train_acc"][-1],
            "final_val_acc": metadata["history"]["val_acc"][-1],
        }
    )

    for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(
        zip(
            metadata["history"]["train_loss"],
            metadata["history"]["val_loss"],
            metadata["history"]["train_acc"],
            metadata["history"]["val_acc"],
        )
    ):
        mlflow.log_metrics(
            {
                "epoch_train_loss": train_loss,
                "epoch_val_loss": val_loss,
                "epoch_train_acc": train_acc,
                "epoch_val_acc": val_acc,
            }
        )

    mlflow.log_artifact(str(METADATA_PATH))

    class RefundClassifer(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import timm
            from torchvision import transforms

            self.device = "cpu"
            checkpoint = torch.load(
                context.artifacts["model"], map_location=self.device
            )

            self.model = timm.create_model(
                "efficientnet_b0", pretrained=False, num_classes=5
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.class_to_idx = checkpoint["class_to_idx"]
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        def predict(self, context, model_input):
            from PIL import Image

            predictions = []
            for img_path in model_input["image_paths"]:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img).unsqueeze(0)

                with torch.no_grad():
                    output = self.model(img_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()

                predictions.append(
                    {
                        "predicted_class": self.idx_to_class[pred_idx],
                        "confidence": float(probs[pred_idx]),
                        "all_probabilities": {
                            self.idx_to_class[i]: float(probs[i])
                            for i in range(len(probs))
                        },
                    }
                )

            return predictions

    artifacts = {"model": str(MODEL_PATH)}
    conda_env = {
        "name": "refund-classifier-env",
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.12",
            "uv",
            {"uv": ["mlflow", "torch", "torchvision", "timm", "pillow"]},
        ],
    }

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=RefundClassifer(),
        artifacts=artifacts,
        conda_env=conda_env,
        registered_model_name="refund-classifier",
    )

    print("Model registered successfully!")
    print(f"  Run ID: {run.info.run_id}")
    print(f"  Model URI: runs:/{run.info.run_id}/model")
