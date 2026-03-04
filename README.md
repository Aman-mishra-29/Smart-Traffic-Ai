#  Smart Traffic AI – Real-Time Vehicle Speed & Violation Detection System <br>
A real-time traffic analytics system built using **YOLOv8**, **ByteTrack**, and **FastAPI** for vehicle detection, tracking, and overspeed violation monitoring.

This project demonstrates how to move from **model training** to a **production-style ML backend system**. <br><br>
**Project Overview** <br>
Smart Traffic AI integrates:

- Deep learning-based object detection (YOLOv8) <br>
- Multi-object tracking (ByteTrack) <br>
- Speed estimation using pixel-to-meter conversion <br>
- Overspeed violation detection <br>
- Thread-safe background processing <br>
- REST API exposure with Swagger documentation <br>

The system processes video streams in real time and exposes traffic analytics via API endpoints. <br><br>
**ML Engineering Highlights**

- Integrated YOLOv8 inference pipeline with backend API service <br>
- Designed modular architecture separating detection, tracking, and analytics layers <br>
- Implemented real-time speed estimation logic <br>
- Built thread-safe analytics state management <br>
- Structured environment-based configuration <br>
- Designed project for scalability and production deployment <br><br>

**System Architecture**

 Video Stream
↓
YOLOv8 Object Detection
↓
ByteTrack Multi-Object Tracking
↓
Speed Estimation (Pixel-to-Meter Conversion)
↓
Overspeed Violation Detection
↓
Thread-Safe Analytics Engine
↓
FastAPI REST API <br><br>
**Tech Stack**

- Python 3.10+ <br>
- YOLOv8 (Ultralytics) <br>
- ByteTrack (Supervision) <br>
- FastAPI <br>
- OpenCV <br>
- NumPy <br>
- Pydantic <br>
- Uvicorn <br>

**Project Structure** <br>
Smart-Traffic-AI/ <br>
│ <br>
├── app/ <br>
│ ├── api/ <br>
│ ├── models/ <br>
│ ├── services/ <br>
│ ├── config.py <br>
│ ├── main.py <br>
│ <br>
├── model/ <br>
│ <br>
├── notebooks/ <br>
│ └── Traffic_prediction.ipynb <br>
│ <br>
├── dataset/ <br>
│ ├── images/ <br>
│ ├── labels/ <br>
│ └── data.yaml <br>
│ <br>
├── requirements.txt <br>
├── .env.example <br>
├── .gitignore <br>
└── README.md <br>

**Dataset** <br>
The dataset follows YOLO format: <br>
dataset/ <br>
├── images/train <br>
├── images/val <br>
├── labels/train <br>
├── labels/val <br>
└── data.yaml <br>

Full dataset is not included due to size limitations. <br><br>

**Model Training**

Model training notebook: <br>
notebooks/Traffic_prediction.ipynb <br><br>

Training includes:

- Dataset preparation <br>
- Model training using YOLOv8 <br>
- Performance evaluation <br>
- Model export for backend inference <br>

**Running the Backend Locally**

**Clone the repository**

**bash**
git clone https://github.com/Aman-mishra-29/Smart-Traffic-AI.git
cd Smart-Traffic-AI <br><br>

**Create Virtual Environment** <br>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate <br>

**Install Dependencies**<br>
pip install -r requirements.txt <br>
Update paths if necessary. <br>

**Create .env File** <br>
Copy from example: <br>
cp .env.example .env <br>

**Run Server** <br>
uvicorn app.main:app --reload <br>

**Open Swagger documentation:** <br>
http://127.0.0.1:8000/docs <br>

**Key Features Implemented** <br>

- Real-time vehicle detection <br>
- Multi-object ID tracking <br>
- Speed estimation algorithm <br>
- Overspeed violation logging <br>
- Thread-safe background processing <br>
- OpenAPI documentation <br>
- Environment-based configuration <br>

**Configuration** <br>
Environment variables are managed via .env. <br>
Example configuration: <br>
MODEL_PATH=models/best.pt <br>
VIDEO_PATH=sample_video.mp4 <br>
PIXELS_PER_METER=8 <br>
FPS=30 <br>
SPEED_LIMIT=60 <br>

**Scalability Considerations** <br>

This project is structured to support: <br>
- Model versioning <br>
- Dataset versioning <br>
- External database integration (PostgreSQL) <br>
- Redis caching <br>
- JWT authentication <br>
- Docker containerization <br>
- Cloud deployment (AWS / Azure / GCP) <br>

**Future Improvements** <br>

- Persistent database storage <br>
- Continuous model retraining pipeline <br>
- ML experiment tracking (MLflow) <br>
- Prometheus monitoring <br>
- Distributed processing <br>

**Author** <br>
Aman Mishra <br>
Aspiring ML Engineer | Data Analyst | AI Enthusiast <br>
