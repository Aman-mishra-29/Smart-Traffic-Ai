import cv2
import threading
import time
import logging
import numpy as np
import supervision as sv

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from app.config import (
    MODEL_PATH,
    VIDEO_PATH,
    PIXELS_PER_METER,
    FPS,
    SPEED_LIMIT
)

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------
# Initialize FastAPI
# -----------------------------
from app.api.routes import router
app = FastAPI(
    title="Smart Traffic AI Backend",
    version="1.0.0"
)
app.include_router(router, prefix="/api")

# -----------------------------
# Load Model & Tracker
# -----------------------------
logger.info("Loading YOLO model...")
model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()

logger.info("Opening video source...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    logger.error("Failed to open video source.")
else:
    logger.info("Video source connected successfully.")

# -----------------------------
# Shared Global State
# -----------------------------
latest_stats = {
    "active_vehicles": 0,
    "average_speed": 0,
    "overspeed_count": 0,
    "total_processed_frames": 0
}

violations = []
vehicle_positions = {}
lock = threading.Lock()

# -----------------------------
# Background Processing Thread
# -----------------------------
def process_video_stream():
    global latest_stats, violations

    logger.info("Starting background traffic processing thread...")

    while True:
        ret, frame = cap.read()

        if not ret:
            logger.warning("Video ended or failed. Restarting stream...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, conf=0.4)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        active_count = 0
        speed_sum = 0
        overspeed_count = 0

        for box, tracker_id in zip(detections.xyxy, detections.tracker_id):

            if tracker_id is None:
                continue

            active_count += 1

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tracker_id in vehicle_positions:
                px, py = vehicle_positions[tracker_id]
                pixel_dist = np.hypot(cx - px, cy - py)
                speed = (pixel_dist / PIXELS_PER_METER) * FPS * 3.6
            else:
                speed = 0

            vehicle_positions[tracker_id] = (cx, cy)
            speed_sum += speed

            if speed > SPEED_LIMIT:
                overspeed_count += 1
                violations.append({
                    "vehicle_id": tracker_id,
                    "speed_kmh": round(speed, 2),
                    "timestamp": time.time()
                })

        avg_speed = round(speed_sum / active_count, 2) if active_count > 0 else 0

        # Thread-safe update
        with lock:
            latest_stats["active_vehicles"] = active_count
            latest_stats["average_speed"] = avg_speed
            latest_stats["overspeed_count"] = overspeed_count
            latest_stats["total_processed_frames"] += 1

        time.sleep(1 / FPS)


# -----------------------------
# FastAPI Startup Event
# -----------------------------
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=process_video_stream, daemon=True)
    thread.start()
    logger.info("Background thread started successfully.")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/api/stats")
def get_stats():
    with lock:
        return JSONResponse(content=latest_stats)



@app.get("/api/health")
def health_check():
    return {
        "status": "running",
        "model_loaded": True,
        "video_connected": cap.isOpened()
    }


@app.get("/")
def root():
    return {
        "message": "Smart Traffic AI Backend is running",
        "docs": "/docs"
    }