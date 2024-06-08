import logging
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir))
)

LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(LOGGER_FORMAT)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logs_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir, "logs")
)
os.makedirs(logs_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(logs_dir, "log.txt"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
