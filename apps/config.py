import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------
# Load .env file (if present)
# -----------------------------------
load_dotenv()

# -----------------------------------
# Base Directory
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------
# Environment Mode
# -----------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

# -----------------------------------
# Model Configuration
# -----------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "best.pt")
)

# -----------------------------------
# Video Source Configuration
# -----------------------------------
VIDEO_PATH = os.getenv(
    "VIDEO_PATH",
    str(BASE_DIR / "data" / "sample_video.mp4")
)

# If you want to use webcam instead:
USE_WEBCAM = os.getenv("USE_WEBCAM", "false").lower() == "true"
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))

# ==============================
# TRACKING SETTINGS
# ==============================

MAX_TRACKED_VEHICLES = 1000

# -----------------------------------
# Traffic Analytics Configuration
# -----------------------------------
PIXELS_PER_METER = float(os.getenv("PIXELS_PER_METER", 8))
FPS = int(os.getenv("FPS", 30))
SPEED_LIMIT = float(os.getenv("SPEED_LIMIT", 60))

# -----------------------------------
# Server Configuration
# -----------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 2))

# -----------------------------------
# Validation Checks
# -----------------------------------
def validate_config():
    errors = []

    if not USE_WEBCAM and not Path(MODEL_PATH).exists():
        errors.append(f"MODEL_PATH does not exist: {MODEL_PATH}")

    if not USE_WEBCAM and not Path(VIDEO_PATH).exists():
        errors.append(f"VIDEO_PATH does not exist: {VIDEO_PATH}")

    if SPEED_LIMIT <= 0:
        errors.append("SPEED_LIMIT must be positive.")

    if FPS <= 0:
        errors.append("FPS must be positive.")

    if errors:
        raise ValueError("Configuration Errors:\n" + "\n".join(errors))


# Run validation at import time
validate_config()