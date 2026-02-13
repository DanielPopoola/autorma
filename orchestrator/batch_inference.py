import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/orchestrator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

settings = get_settings()


INPUT_DIR = Path("data/inference/input")
OUTPUT_DIR = Path("data/inference/output")
CHECKPOINT_DIR = Path("data/inference/checkpoints")


class BatchOrchestrator:
    def __init__(self):
        self.checkpoint_file = CHECKPOINT_DIR / "checkpoint.json"
        self.failed_images = []

    def load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"processed_images": [], "last_run": None}

    def save_checkpoint(self, checkpoint: dict) -> None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def get_unprocessed_images(self, checkpoint: dict) -> list[Path]:
        all_images = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
        processed = set(checkpoint["processed_images"])
        return [img for img in all_images if img.name not in processed]

    def predict_batch(self, image_paths: list[Path]) -> list[dict]:
        try:
            response = requests.post(
                f"{settings.MODEL_SERVICE_URL}/predict",
                json={"image_paths": [str(p.absolute()) for p in image_paths]},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["predictions"]
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    def process_batch(self, batch: list[Path], batch_num: int) -> list[dict]:
        logger.info(f"Processing batch {batch_num} ({len(batch)} images)")

        results = []
        try:
            predictions = self.predict_batch(batch)

            for img_path, pred in zip(batch, predictions):
                results.append(
                    {
                        "image_name": img_path.name,
                        "image_path": str(img_path),
                        "predicted_class": pred["predicted_class"],
                        "confidence": pred["confidence"],
                        "probabilities": pred["all_probabilities"],
                        "batch_number": batch_num,
                        "processed_at": datetime.now().isoformat(),
                    }
                )

            logger.info(f"✓ Batch {batch_num} completed successfully")
            return results

        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            for img in batch:
                self.failed_images.append(
                    {
                        "image_name": img.name,
                        "batch_number": batch_num,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            return []

    def run(self):
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting batch inference")

        # Check model service health
        try:
            health = requests.get(f"{settings.MODEL_SERVICE_URL}/health", timeout=5).json()
            logger.info(f"Model service: {health}")
        except Exception as e:
            logger.error(f"Model service unavailable: {e}")
            return

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        logger.info(
            f"Loaded checkpoint: {len(checkpoint['processed_images'])} already processed"
        )

        # Get unprocessed images
        unprocessed = self.get_unprocessed_images(checkpoint)
        if not unprocessed:
            logger.info("No new images to process")
            return

        logger.info(f"Found {len(unprocessed)} new images to process")

        # Process in batches
        all_results = []
        batches = [
            unprocessed[i : i + settings.BATCH_SIZE]
            for i in range(0, len(unprocessed), settings.BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches, 1):
            batch_results = self.process_batch(batch, batch_num)
            all_results.extend(batch_results)

            # Update checkpoint after each batch
            checkpoint["processed_images"].extend([img.name for img in batch])
            checkpoint["last_run"] = datetime.now().isoformat()
            self.save_checkpoint(checkpoint)
            logger.info(
                f"Checkpoint updated: {len(checkpoint['processed_images'])} total processed"
            )

        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = OUTPUT_DIR / f"predictions_{timestamp}.json"

        output = {
            "run_timestamp": timestamp,
            "total_images": len(unprocessed),
            "successful": len(all_results),
            "failed": len(self.failed_images),
            "duration_seconds": round(time.time() - start_time, 2),
            "predictions": all_results,
            "failed_images": self.failed_images,
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        from collections import Counter
        class_dist = Counter([r['predicted_class'] for r in all_results])
        
        from metrics_pusher import MetricsPusher
        pusher = MetricsPusher()
        pusher.push_metrics(
            duration=output['duration_seconds'],
            total=len(unprocessed),
            successful=len(all_results),
            failed=len(self.failed_images),
            class_distribution=dict(class_dist)
        )
        
        logger.info("=" * 60)
        logger.info("✓ Batch inference complete")
        logger.info(f"  Total: {len(unprocessed)} images")
        logger.info(f"  Success: {len(all_results)}")
        logger.info(f"  Failed: {len(self.failed_images)}")
        logger.info(f"  Duration: {output['duration_seconds']}s")
        logger.info(f"  Results saved to: {results_file}")
        logger.info("=" * 60)


if __name__ == "__main__":
    orchestrator = BatchOrchestrator()
    orchestrator.run()
