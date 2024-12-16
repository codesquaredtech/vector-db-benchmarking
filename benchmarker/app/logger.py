import datetime
import logging
from multiprocessing import get_logger

logger = get_logger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(asctime)s| %(levelname)s| %(processName)s] %(message)s"
)
current_datetime = date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
handler = logging.FileHandler(f"/benchmarker/logs/{current_datetime}_selfiender.log")
handler.setFormatter(formatter)

if not len(logger.handlers):
    logger.addHandler(handler)


def get_logger():
    return logger
