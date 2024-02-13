import datetime
import logging
from typing import Mapping

import google.cloud.logging
from google.cloud.logging import _helpers
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging.handlers.transports import background_thread
from google.cloud.logging.handlers.transports.base import Transport
from google.cloud.logging.handlers.transports.sync import SyncTransport

__all__ = ["add_handler", "nodefaults", "FORMAT"]


class CustomLoggingWorker(background_thread._Worker):
    def encode(self, record):
        return {k: str(v) for k, v in record.__dict__.items()}

    def enqueue(
        self, record, message, resource=None, labels=None, trace=None, span_id=None
    ):
        entry = {
            "info": {
                "message": record.msg,
                "python_logger": record.name,
                "record": {k: str(v) for k, v in record.__dict__.items()},
            },
            "severity": record.levelname,
            "resource": resource,
            "labels": labels,
            "trace": trace,
            "span_id": span_id,
            "timestamp": datetime.datetime.utcfromtimestamp(record.created),
        }
        self._queue.put_nowait(entry)


class CustomLoggingTransport(Transport):
    def __init__(
        self,
        client,
        name,
        grace_period=background_thread._DEFAULT_GRACE_PERIOD,
        batch_size=background_thread._DEFAULT_MAX_BATCH_SIZE,
        max_latency=background_thread._DEFAULT_MAX_LATENCY,
    ):
        self.client = client
        logger = self.client.logger(name)
        self.worker = CustomLoggingWorker(
            logger,
            grace_period=grace_period,
            max_batch_size=batch_size,
            max_latency=max_latency,
        )
        self.worker.start()

    def send(
        self, record, message, resource=None, labels=None, trace=None, span_id=None
    ):
        self.worker.enqueue(
            record,
            message,
            resource=resource,
            labels=labels,
            trace=trace,
            span_id=span_id,
        )

    def flush(self):
        self.worker.flush()


class CustomLoggingTransportSync(SyncTransport):
    def __init__(self, client, name):
        super().__init__(client, name)
        self.logger = client.logger(name)

    def send(
        self, record, message, resource=None, labels=None, trace=None, span_id=None
    ):
        info = {
            "message": record.msg,
            "python_logger": record.name,
            "record": {k: str(v) for k, v in record.__dict__.items()},
        }
        self.logger.log_struct(
            info,
            severity=_helpers._normalize_severity(record.levelno),
            resource=resource,
            labels=labels,
            trace=trace,
            span_id=span_id,
        )


class CustomCloudLoggingHandler(CloudLoggingHandler):
    def emit(self, record):
        self.transport.send(record, resource=self.resource, labels=self.labels)


# helper functions
FORMAT = "[%(asctime)s %(levelname)s %(name)s] - %(message)s"


def nodefaults():
    defaults = (
        "google.cloud",
        "google.auth",
        "google_auth_httplib2",
        "urllib3.connectionpool",
        "requests",
        "urllib3",
    )
    for name in defaults:
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(logging.NullHandler())


def add_handler(
    logger: logging.Logger, name: str, labels: Mapping[str, object], mt=False
):
    logger.addHandler(
        CloudLoggingHandler(
            google.cloud.logging.Client(),
            name=name,
            transport=CustomLoggingTransportSync if mt else CustomLoggingTransport,
            labels=labels,
        )
    )
    return logging.LoggerAdapter(logger, labels)
