import logging, os
from datetime import date

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.abspath(os.path.join(base_dir, "../logs"))


filename = os.path.join(log_dir, f"image-insertion-{date.today()}-logs.txt")

file_handler = logging.FileHandler(filename, "a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

file_formater = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formater)


logger.addHandler(console_handler)
logger.addHandler(file_handler)
