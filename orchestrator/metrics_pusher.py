from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import logging

logger = logging.getLogger(__name__)


class MetricsPusher:
    def __init__(self, pushgateway_url="localhost:9091"):
        self.pushgateway_url = pushgateway_url
        self.registry = CollectorRegistry()

        # Define metrics
        self.batch_duration = Gauge(
            "batch_duration_seconds",
            "Batch processing duration",
            registry=self.registry,
        )
        self.images_processed = Gauge(
            "batch_images_processed",
            "Images processed in batch",
            registry=self.registry,
        )
        self.images_failed = Gauge(
            "batch_images_failed", "Images failed in batch", registry=self.registry
        )
        self.batch_success_rate = Gauge(
            "batch_success_rate", "Batch success rate", registry=self.registry
        )

        # Per-class gauges
        self.class_counts = {}

    def add_class_metric(self, class_name: str, count: int):
        if class_name not in self.class_counts:
            self.class_counts[class_name] = Gauge(
                f"predictions_class_{class_name.replace(' ', '_')}",
                f"Predictions for {class_name}",
                registry=self.registry,
            )
        self.class_counts[class_name].set(count)

    def push_metrics(
        self,
        duration: float,
        total: int,
        successful: int,
        failed: int,
        class_distribution: dict,
    ):
        self.batch_duration.set(duration)
        self.images_processed.set(successful)
        self.images_failed.set(failed)

        if total > 0:
            self.batch_success_rate.set(successful / total)

        for class_name, count in class_distribution.items():
            self.add_class_metric(class_name, count)

        try:
            push_to_gateway(
                self.pushgateway_url, job="batch_orchestrator", registry=self.registry
            )
            logger.info("✓ Metrics pushed to Pushgateway")
        except Exception as e:
            logger.warning(f"Failed to push metrics: {e}")
