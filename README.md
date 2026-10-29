# Smart Traffic AI — Real-Time Vehicle Detection & Speed Monitoring

A computer vision and traffic analytics project built using **YOLOv8, ByteTrack, OpenCV, and FastAPI**. The system processes video frames to detect and track vehicles, estimate their speeds, identify potential overspeeding events, and expose traffic statistics through REST APIs.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Speed Estimation Method](#speed-estimation-method)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [Limitations and Known Setup Issues](#limitations-and-known-setup-issues)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## Overview

Smart Traffic AI explores how computer vision and machine learning can be applied to traffic monitoring.

Traditional traffic monitoring often requires manual observation or dedicated hardware. This project demonstrates a software-based approach in which a trained object detection model identifies vehicles in video frames, a multi-object tracker assigns persistent IDs across frames, and a speed estimation algorithm calculates approximate vehicle speeds.

The backend exposes monitoring and analytics information through FastAPI endpoints, making the processing pipeline accessible to API clients.

The current implementation focuses primarily on:

1. Vehicle detection using YOLOv8.
2. Multi-object tracking using ByteTrack.
3. Approximate vehicle speed estimation.
4. Overspeed event detection.
5. Traffic statistics and violation retrieval through REST APIs.

## Key Features

### 1. Vehicle Detection

The system uses a YOLOv8 model to detect vehicles in incoming video frames.

- Processes frames using OpenCV.
- Runs object detection with a configurable confidence threshold in the modular detection service.
- Converts model results into a standardized detections format using Supervision.

The repository includes a trained model checkpoint at `model/best.pt`.

### 2. Multi-Object Tracking

ByteTrack is used to associate detections across successive frames.

- Assigns tracking IDs to detected objects.
- Helps maintain object identity across frames.
- Provides the tracking information required for speed estimation and traffic analytics.

The tracking service uses a lock to protect access to shared tracker state.

### 3. Vehicle Speed Estimation

The application estimates vehicle speed by comparing the vehicle's position in consecutive frames.

The current implementation uses bounding-box center coordinates and a pixel-to-meter conversion factor to approximate speed in kilometres per hour.

The accuracy of this estimate depends on camera placement, calibration, frame rate, and video conditions.

### 4. Overspeed Detection

The analytics logic compares estimated vehicle speeds with a configured speed limit.

When the estimated speed exceeds the configured threshold, the system records an overspeed event in its in-memory analytics state.

Violation records can include:

- Vehicle tracking ID.
- Estimated speed in kilometres per hour.
- Configured speed limit.
- Event timestamp.

**Important:** These are software-generated estimates. They are not independently verified measurements suitable for automatic legal enforcement.

### 5. Traffic Analytics

The API is designed to expose traffic monitoring information, including:

- Active vehicle count.
- Average estimated speed.
- Overspeed count.
- Total tracked vehicles.
- Recent violation events.
- Tracker health information.

Analytics and violations are currently maintained in application memory rather than a persistent database.

### 6. FastAPI REST API

FastAPI provides an API interface for accessing the traffic monitoring system.

- JSON-based responses.
- Health-check endpoints.
- Query parameters for limiting violation results.
- Request testing through Swagger UI.
- Logging and structured endpoint definitions.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Object detection | YOLOv8 |
| Model framework | Ultralytics |
| Object tracking | ByteTrack |
| Detection and tracking utilities | Supervision |
| Video processing | OpenCV |
| Numerical calculations | NumPy |
| Backend framework | FastAPI |
| API server | Uvicorn |
| Request and response validation | Pydantic |
| Environment configuration | python-dotenv |
| Model training | Jupyter Notebook |
| Dataset annotation format | YOLO |

## System Architecture

The application separates video processing, detection, tracking, analytics, and API access into logical components.

```text
               VIDEO SOURCE
                     |
                     v
             OpenCV Frame Reader
                     |
                     v
              YOLOv8 Detection
                     |
                     v
             ByteTrack Tracking
                     |
                     v
            Vehicle Tracking IDs
                     |
                     v
          Speed Estimation Logic
                     |
                     v
          Overspeed Event Analysis
                     |
                     v
          Traffic Analytics State
                     |
                     v
               FastAPI REST API
                     |
                     v
             API Client / Swagger
```

### Processing Workflow

1. Read a frame from the configured video source.
2. Run YOLOv8 inference to detect objects.
3. Convert the detection results into the format used by Supervision.
4. Update the tracker to associate detections with vehicle IDs.
5. Compare vehicle positions across frames.
6. Calculate approximate speed using the configured calibration factor and frame rate.
7. Compare estimated speed against the configured limit.
8. Update traffic statistics and record overspeed events.
9. Make available analytics through the API.

## Project Structure

The repository currently contains the following main files and directories:

```text
Smart-Traffic-Ai/
│
├── apps/
│   ├── api/
│   │   └── routes.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── analytics_service.py
│   │   ├── detection_service.py
│   │   └── tracking_service.py
│   ├── config.py
│   └── main.py
│
├── dataset/
│   ├── data.yaml
│   ├── images/
│   └── lables/
│
├── model/
│   └── best.pt
│
├── notebooks/
│   └── Traffic_prediction.ipynb
│
├── .env.example
├── .gitignore
├── requirments.txt
└── README.md
```

**Note:** The current repository uses `apps/`, but the Python source files import modules through `app.*`. This naming mismatch must be resolved before the setup commands below will work. The current dependency filename is also misspelled as `requirments.txt`.

## Prerequisites

Install the following before running the project:

- **Python 3.10** — the recommended version based on the dependency versions.
- **pip** — Python package installer.
- **Git** — to clone the repository.
- **A compatible CPU or NVIDIA GPU** — GPU acceleration may improve model inference performance.
- **A valid video file** — used as the traffic input source.
- **Available memory and disk space** — model loading and video processing require sufficient system resources.

The tracked model checkpoint is included in the repository. A complete training dataset and a verified working video sample are not included in the current repository structure.

## Installation and Setup

### Step 1: Clone the repository

Open PowerShell or a terminal and run:

```powershell
git clone https://github.com/Aman-mishra-29/Smart-Traffic-Ai.git
cd Smart-Traffic-Ai
```

### Step 2: Resolve the Python package naming mismatch

The code currently imports modules such as `app.config`, but the repository folder is named `apps`.

The simplest approach is to rename `apps` to `app` so the folder matches the existing imports.

In PowerShell, from the repository root:

```powershell
Rename-Item apps app
```

The project also contains files named `_init_.py`. Rename these to `__init__.py` for the conventional Python package layout:

```powershell
Rename-Item app\_init_.py __init__.py
Rename-Item app\api\_init_.py __init__.py
Rename-Item app\services\_init_.py __init__.py
```

These are local source-structure fixes. Commit the corresponding changes if you want other users to clone the corrected repository directly.

### Step 3: Create a virtual environment

From the repository root:

```powershell
python -m venv venv
```

Activate it in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### Step 4: Install dependencies

The dependency file is currently named `requirments.txt`, so use:

```powershell
python -m pip install --upgrade pip
pip install -r requirments.txt
```

If you rename it to `requirements.txt`, use that corrected filename in this command and update the repository accordingly.

PyTorch installation can vary depending on whether you intend to use a CPU or a compatible NVIDIA GPU. Follow the official PyTorch installation instructions if the pinned version does not install correctly on your machine.

## Configuration

The application reads configuration values from environment variables through `apps/config.py` (or `app/config.py` after the folder is renamed).

The root `.env.example` file provides example settings.

Copy it to a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and ensure the model and video paths point to files that actually exist.

Example for this repository:

```dotenv
MODEL_PATH=model/best.pt
VIDEO_PATH=C:/traffic-data/sample_video.mp4

CONFIDENCE_THRESHOLD=0.4
PIXELS_PER_METER=8
FPS=30
SPEED_LIMIT=60

HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
```

The video path above is an example. Replace it with the actual path to your own valid traffic video.

### Configuration Parameters

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Path to the trained YOLO model checkpoint |
| `VIDEO_PATH` | Path to the input video |
| `CONFIDENCE_THRESHOLD` | Detection confidence threshold used by the modular detection service |
| `PIXELS_PER_METER` | Pixel-to-meter conversion factor used by speed estimation |
| `FPS` | Frame-rate value used for speed estimation and processing configuration |
| `SPEED_LIMIT` | Overspeed threshold in km/h |
| `HOST` | Configured server host |
| `PORT` | Configured server port |
| `LOG_LEVEL` | Intended logging level setting |

The current configuration has a few inconsistencies that need attention:

- The default model path is `models/best.pt`, while the tracked checkpoint is under `model/best.pt`.
- The default video path is `data/sample_video.mp4`, while the tracked sample path is under `dataset/images/sample/sample_video.mp4`.
- The tracked sample-video entry is not a usable video file, so provide a valid video locally.
- The current `main.py` reads the video using `cv2.VideoCapture(VIDEO_PATH)`. Although `USE_WEBCAM` appears in the configuration file, the current main processing loop does not use it.

The paths must be fixed before starting the backend because the configuration validates the model and video paths during import.

## Running the Application

### Step 1: Confirm the model and video paths

From the project root, ensure your `.env` contains valid paths for:

- `MODEL_PATH`
- `VIDEO_PATH`

For example, `model/best.pt` should point to the tracked checkpoint, while `VIDEO_PATH` should point to an actual video file on your machine.

### Step 2: Start the FastAPI backend

After resolving the package naming issue, activating the virtual environment, installing the dependencies, and configuring valid paths, run:

```powershell
uvicorn app.main:app --reload
```

If the server starts successfully, the API should be available at:

- **API root:** http://127.0.0.1:8000/
- **Health check:** http://127.0.0.1:8000/health
- **Interactive API documentation:** http://127.0.0.1:8000/docs
- **Alternative API documentation:** http://127.0.0.1:8000/redoc

The backend loads the YOLO model and opens the configured video source. The video-processing loop starts in a background thread when the FastAPI application starts.

Keep the terminal open while the service runs.

## API Endpoints

The current source defines the following API operations.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Returns backend service information |
| GET | `/health` | Returns basic backend and video-source status |
| GET | `/api/` | Returns API router information |
| GET | `/api/stats` | Returns traffic statistics |
| GET | `/api/violations` | Returns recent overspeed events |
| GET | `/api/health` | Returns tracker-related health information |
| POST | `/api/admin/reset` | Resets analytics state |

The violations endpoint supports a `limit` query parameter between 1 and 100, with a default of 20.

Example:

```text
http://127.0.0.1:8000/api/violations?limit=10
```

### Example: Health Check

Request:

```http
GET /health
```

An example response when the backend is running:

```json
{
  "status": "running",
  "model_loaded": true,
  "video_connected": true
}
```

This response represents the basic health-check structure in the current application.

### Example: Traffic Statistics

Request:

```http
GET /api/stats
```

The intended response includes fields such as active vehicle count, average estimated speed, overspeed count, and processed-frame or tracking statistics, depending on which registered route handles the request.

**Implementation note:** Both `main.py` and the included API router currently define overlapping statistics and health routes. Their state sources are also separate. Consolidate these definitions before relying on the endpoints for consistent live statistics.

Use Swagger UI at `http://127.0.0.1:8000/docs` to inspect the actual registered paths and test them.

## Speed Estimation Method

The current speed estimation logic uses object positions in consecutive frames.

Let:

- \(d_p\) be the pixel distance between two observed positions.
- \(P\) be the configured number of pixels per meter.
- \(F\) be the configured frame rate.

The approximate distance in meters is:

\[
d_m = \frac{d_p}{P}
\]

The approximate speed in kilometres per hour is:

\[
v_{\text{km/h}} = \frac{d_p}{P} \times F \times 3.6
\]

This calculation is a simplified estimate. It assumes a suitable relationship between image-plane displacement and real-world distance.

For more reliable measurements, the camera geometry, road plane, perspective effects, frame timing, and calibration factor must be validated for the specific camera and road.

## Model Training

The repository contains a Jupyter notebook:

```text
notebooks/Traffic_prediction.ipynb
```

The notebook is the starting point for examining the model-training workflow.

The repository also contains:

```text
model/best.pt
dataset/data.yaml
```

The trained checkpoint is used for inference. The dataset configuration identifies a YOLO-format vehicle detection dataset.

To reproduce training, you will need the complete dataset and a suitable environment for the pinned model framework. The current repository does not include a complete training dataset, so training may require downloading or separately configuring the original data.

## Dataset

The repository's `dataset/data.yaml` declares:

- One class: `Vehicle`.
- Training and validation paths.
- A Roboflow dataset reference.

The declared class configuration is:

```yaml
nc: 1
names: ['Vehicle']
```

The referenced dataset is available at:

[Roboflow — Vehicle Detection YOLOv8 Dataset](https://universe.roboflow.com/farzad/vehicle_detection_yolov8/dataset/3)

Review that dataset's licence and terms before reusing or redistributing its content.

The complete dataset is not included in this repository. Confirm that the configured training and validation directories exist before running the notebook.

## Limitations and Known Setup Issues

The current repository is a development-stage project. The following points should be reviewed before using it outside a local experiment.

1. **Python package naming:** The code imports `app.*`, but the repository folder is currently named `apps`.
2. **Dependency filename:** The requirements file is named `requirments.txt`.
3. **Model path:** The default configuration points to `models/best.pt`, while the tracked model is `model/best.pt`.
4. **Video source:** A valid video must be provided; the tracked sample-video entry is not a usable video.
5. **Statistics routes:** Some API endpoints are defined in both `main.py` and `api/routes.py`.
6. **Analytics state:** Processing state and violations are held in memory and are lost when the process restarts.
7. **Speed calibration:** The pixel-to-meter method is approximate and requires camera-specific calibration.
8. **Security:** The reset endpoint currently has no authentication, and the main application configures permissive CORS. Restrict access before exposing the service on a public network.
9. **Real-world enforcement:** The project does not independently verify speed measurements and should not be used as an automatic enforcement system without substantial validation.

## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

The folder and imports do not match. Rename `apps/` to `app/`, or update the imports consistently across the project.

### Model file not found

Check the value of `MODEL_PATH` in `.env`. The tracked checkpoint is at:

```text
model/best.pt
```

### Video file cannot be opened

Check the `VIDEO_PATH` setting and confirm that the video exists and can be read by OpenCV.

### API starts but statistics do not update as expected

Review the overlapping route definitions and confirm that the endpoint reads from the same analytics state that the video-processing loop updates.

### Package installation fails

Confirm you are using a supported Python version, activate the virtual environment, and check the pinned PyTorch and Ultralytics dependencies. GPU and CPU installations may require different setup steps.

### Swagger UI does not open

Check the backend terminal for startup errors, then visit:

```text
http://127.0.0.1:8000/docs
```

## Future Enhancements

Potential improvements include:

- Camera calibration and more robust speed estimation.
- Persistent storage for traffic statistics and violation history.
- Improved video-source selection and webcam support.
- More comprehensive tests for detection, tracking, and analytics.
- Consolidation of duplicate API routes and shared state.
- Authentication and authorization for administrative endpoints.
- More complete dataset and model versioning.
- Reproducible training and model-evaluation pipelines.
- Monitoring, structured deployment configuration, and containerization.
- Additional traffic violations, such as red-light or wrong-lane violations, if the corresponding detection logic is implemented and validated.

These are potential extensions, not claims that those features are currently implemented.

## Author

**Aman Mishra**

- GitHub: [Aman-mishra-29](https://github.com/Aman-mishra-29)
- Project: [Smart Traffic AI](https://github.com/Aman-mishra-29/Smart-Traffic-Ai)
- LinkedIn: [Aman Mishra](https://www.linkedin.com/in/aman-mishra-479973281)

---

If you encounter a problem, check the configuration and terminal logs first. When reporting an issue, include the steps to reproduce it and the relevant error message, but remove credentials and private information before sharing logs.
