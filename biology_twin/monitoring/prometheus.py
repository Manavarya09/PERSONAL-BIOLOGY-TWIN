from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import random

# Metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('biology_twin_requests_total', 'Total requests', ['method', 'endpoint'])
MODEL_INFERENCE_TIME = Summary('model_inference_seconds', 'Time for model inference')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')

class Monitoring:
    """Prometheus monitoring for the system."""

    def __init__(self, port: int = 8001):
        # self.port = port
        # start_http_server(port)
        # print(f"Monitoring server started on port {port}")
        pass

    @REQUEST_TIME.time()
    def process_request(self, method: str, endpoint: str):
        """Decorator for request processing."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        # Simulate processing
        time.sleep(random.uniform(0.1, 0.5))

    @MODEL_INFERENCE_TIME.time()
    def model_inference(self):
        """Time model inference."""
        time.sleep(random.uniform(0.05, 0.2))

    def update_active_users(self, count: int):
        """Update active users gauge."""
        ACTIVE_USERS.set(count)