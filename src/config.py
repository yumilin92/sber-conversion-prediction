import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.getenv("MODELS_DIR", ROOT / "models"))
STORE = Path(os.getenv("STORE_DIR", ROOT / "store"))
PROCESSED = Path(os.getenv("DATA_DIR", ROOT / "data" / "processed"))
