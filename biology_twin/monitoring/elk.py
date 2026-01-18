import logging
from elasticsearch import Elasticsearch
import json
from datetime import datetime

class ELKLogger:
    """ELK stack logging for the system."""

    def __init__(self, es_host: str = "localhost", es_port: int = 9200, index: str = "biology-twin-logs"):
        self.es = Elasticsearch(f"http://{es_host}:{es_port}")
        self.index = index
        self.logger = logging.getLogger("biology_twin")
        self.logger.setLevel(logging.INFO)
        
        # Elasticsearch handler
        es_handler = ElasticsearchHandler(self.es, self.index)
        es_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        es_handler.setFormatter(formatter)
        self.logger.addHandler(es_handler)

    def log_request(self, method: str, endpoint: str, user_id: str = None, duration: float = None):
        """Log API request."""
        self.logger.info(f"Request: {method} {endpoint}", extra={
            "user_id": user_id,
            "duration": duration,
            "type": "request"
        })

    def log_model_inference(self, model_name: str, input_shape: tuple, output_shape: tuple, time_taken: float):
        """Log model inference."""
        self.logger.info(f"Model inference: {model_name}", extra={
            "input_shape": input_shape,
            "output_shape": output_shape,
            "time_taken": time_taken,
            "type": "inference"
        })

    def log_error(self, error: str, traceback: str = None):
        """Log error."""
        self.logger.error(f"Error: {error}", extra={
            "traceback": traceback,
            "type": "error"
        })


class ElasticsearchHandler(logging.Handler):
    """Custom logging handler for Elasticsearch."""

    def __init__(self, es_client, index):
        super().__init__()
        self.es = es_client
        self.index = index

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "extra": getattr(record, "extra", {})
            }
            self.es.index(index=self.index, body=log_entry)
        except Exception:
            self.handleError(record)