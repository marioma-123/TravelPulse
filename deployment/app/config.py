import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"

ARTIFACTS_FILE = "artifacts.json"     
LANDMARKS_FILE = "landmarks_clean.csv"

CITY_CENTERS = {
    "Amsterdam": (52.3676, 4.9041),
    "Barcelona": (41.3874, 2.1686),
    "Berlin": (52.5200, 13.4050),
    "Columbus": (39.9612, -82.9988),
}

MAX_DEGREES_FROM_CENTER = 1.0


def get_model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
