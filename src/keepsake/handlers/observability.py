"""Structured JSON logging for Lambda via AWS Lambda Powertools.

Library modules log with the standard `logging` module under the `keepsake` logger. Inside
Lambda, the Powertools JSON formatter is copied onto that logger so every record (including its
`extra` fields) is emitted as one JSON line with the Lambda request id. Outside Lambda (tests,
the CLI) logging is left untouched.
"""

from __future__ import annotations

import logging
import os

from aws_lambda_powertools import Logger
from aws_lambda_powertools.logging import utils

LOGGER = Logger(service=os.environ.get("POWERTOOLS_SERVICE_NAME", "keepsake"))


def configure_logging() -> None:
    if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return
    logging.getLogger("keepsake").setLevel(logging.INFO)
    utils.copy_config_to_registered_loggers(source_logger=LOGGER, include={"keepsake"})


configure_logging()
