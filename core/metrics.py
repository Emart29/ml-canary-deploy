from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

__all__ = [
    "CanaryMetrics", "METRICS_REGISTRY", "generate_metrics_output",
    "CONTENT_TYPE_LATEST", "LATENCY_BUCKETS",
]

# Shared registry for the whole app — avoids clashing with the default global registry
# when multiple services run in the same interpreter / process.
METRICS_REGISTRY = CollectorRegistry()

LATENCY_BUCKETS = (1, 5, 10, 25, 50, 100, 200, 500, 1000, 2500, 5000)


class CanaryMetrics:
    """Singleton-style metrics holder. Instantiate once at app startup and
    inject everywhere predictions are served."""

    def __init__(self, registry: CollectorRegistry = METRICS_REGISTRY):
        self._registry = registry

        self.prediction_requests_total = Counter(
            "prediction_requests_total",
            "Total prediction requests served",
            labelnames=("deployment", "model_version", "model_role", "status"),
            registry=registry,
        )
        self.prediction_latency_ms = Histogram(
            "prediction_latency_ms",
            "Prediction latency in milliseconds",
            labelnames=("deployment", "model_version", "model_role"),
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self.canary_traffic_pct = Gauge(
            "canary_traffic_pct",
            "Current canary traffic percentage",
            labelnames=("deployment",),
            registry=registry,
        )
        self.model_accuracy = Gauge(
            "model_accuracy",
            "Training accuracy of the deployed model",
            labelnames=("deployment", "model_version", "model_role"),
            registry=registry,
        )

    # ------------------------------------------------------------------
    # Hot-path recording
    # ------------------------------------------------------------------

    def record_prediction(
        self, deployment_name: str, model_version_str: str,
        model_role: str, latency_ms: float, is_error: bool,
    ) -> None:
        status = "error" if is_error else "success"
        self.prediction_requests_total.labels(
            deployment=deployment_name, model_version=model_version_str,
            model_role=model_role, status=status,
        ).inc()
        if not is_error:
            self.prediction_latency_ms.labels(
                deployment=deployment_name, model_version=model_version_str,
                model_role=model_role,
            ).observe(latency_ms)

    def update_traffic_split(self, deployment_name: str, canary_pct: float) -> None:
        self.canary_traffic_pct.labels(deployment=deployment_name).set(canary_pct)

    def set_model_accuracy(
        self, deployment_name: str, model_version_str: str,
        model_role: str, accuracy: float,
    ) -> None:
        self.model_accuracy.labels(
            deployment=deployment_name, model_version=model_version_str,
            model_role=model_role,
        ).set(accuracy)

    # ------------------------------------------------------------------
    # Reading current values back out of the registry
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile_from_buckets(buckets: list[tuple[float, float]], total: float, q: float) -> float:
        """Linear-interpolated percentile from cumulative histogram buckets.
        `buckets` is a list of (upper_bound, cumulative_count) sorted ascending."""
        if total <= 0:
            return 0.0
        target = q * total
        prev_le = 0.0
        prev_count = 0.0
        for le, cum in buckets:
            if cum >= target:
                if le == float("inf"):
                    return prev_le
                bucket_count = cum - prev_count
                if bucket_count <= 0:
                    return le
                frac = (target - prev_count) / bucket_count
                return prev_le + frac * (le - prev_le)
            if le != float("inf"):
                prev_le = le
            prev_count = cum
        return prev_le

    def _role_stats(self, deployment_name: str, model_role: str) -> dict:
        request_count = 0.0
        error_count = 0.0
        buckets_by_le: dict[float, float] = {}
        latency_count = 0.0

        for family in self._registry.collect():
            for sample in family.samples:
                labels = sample.labels
                if labels.get("deployment") != deployment_name:
                    continue
                if labels.get("model_role") != model_role:
                    continue

                if sample.name == "prediction_requests_total":
                    request_count += sample.value
                    if labels.get("status") == "error":
                        error_count += sample.value
                elif sample.name == "prediction_latency_ms_bucket":
                    le = float(labels.get("le", "inf"))
                    buckets_by_le[le] = buckets_by_le.get(le, 0.0) + sample.value
                elif sample.name == "prediction_latency_ms_count":
                    latency_count += sample.value

        sorted_buckets = sorted(buckets_by_le.items(), key=lambda kv: kv[0])
        error_rate = (error_count / request_count) if request_count > 0 else 0.0
        return {
            "request_count": int(request_count),
            "error_count": int(error_count),
            "error_rate": round(error_rate, 4),
            "latency_p50_ms": round(self._percentile_from_buckets(sorted_buckets, latency_count, 0.50), 2),
            "latency_p95_ms": round(self._percentile_from_buckets(sorted_buckets, latency_count, 0.95), 2),
            "latency_p99_ms": round(self._percentile_from_buckets(sorted_buckets, latency_count, 0.99), 2),
        }

    def _read_traffic_pct(self, deployment_name: str) -> float:
        for family in self._registry.collect():
            if family.name != "canary_traffic_pct":
                continue
            for sample in family.samples:
                if sample.labels.get("deployment") == deployment_name:
                    return float(sample.value)
        return 0.0

    def get_metrics_snapshot(self, deployment_name: str) -> dict:
        return {
            "baseline": self._role_stats(deployment_name, "baseline"),
            "canary": self._role_stats(deployment_name, "canary"),
            "canary_traffic_pct": self._read_traffic_pct(deployment_name),
        }


def generate_metrics_output() -> bytes:
    """Prometheus exposition format for the /metrics endpoint."""
    return generate_latest(METRICS_REGISTRY)
