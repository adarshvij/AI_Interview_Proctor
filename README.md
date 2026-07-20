<div align="center">

# 🛡️ AI Interview Proctor System

**Real-Time AI-Powered Monitoring for Online Interviews**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00A98F?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-0080FF?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

An intelligent proctoring system that uses **computer vision** and **deep learning** to detect suspicious activities — such as multiple faces, phone usage, and gaze deviations — during remote assessments. Built for real-time performance on standard hardware.

[Features](#-features) · [Architecture](#-system-architecture) · [Installation](#-installation) · [Usage](#-usage) · [Optimization](#-performance-optimization)

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)

- [Performance Optimization](#-performance-optimization)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)

---

## 🔍 About the Project

Remote interviews and online assessments are increasingly common, but maintaining integrity without physical supervision remains a significant challenge. **AI Interview Proctor** addresses this by providing automated, real-time monitoring through a single webcam feed.

The system runs a multi-stage computer vision pipeline — face detection, gaze tracking, and object detection — to identify potential malpractice. It aggregates violations into a weighted risk score, provides instant visual feedback through an overlay and live dashboard, and generates downloadable session reports for reviewers. The project leverages pretrained MediaPipe and YOLOv8 models rather than custom-trained networks, placing its emphasis on real-time computer vision engineering, system-level performance optimization, and end-to-end integration.

**Key design goals:**
- **Real-time performance** on consumer-grade laptops (no GPU required)
- **Modular architecture** with independently testable detector components
- **Production-quality UI** with a live Streamlit dashboard

---

## ✨ Features

| Category | Feature | Description |
|----------|---------|-------------|
| 👤 **Face Detection** | Presence Verification | Confirms that the candidate's face is visible throughout the session |
| 👥 **Multi-Face Detection** | Crowd Detection | Flags when more than one face appears in the camera frame |
| 📱 **Phone Detection** | Device Recognition | Identifies cell phones in the frame using YOLOv8 object detection |
| 👁️ **Gaze Tracking** | Head Pose Estimation | Monitors horizontal (yaw) and vertical (pitch) head orientation |
| 🔄 **Looking-Away Detection** | Attention Monitoring | Flags when the candidate looks left, right, or down away from the screen |
| ⚠️ **Risk Scoring** | Weighted Aggregation | Computes a cumulative risk score with severity-based violation weights |
| 📊 **Live Dashboard** | Real-Time Metrics | Displays FPS, risk level, violation counters, and gaze direction |
| 📄 **Report Generation** | Session Summary | Exports a timestamped CSV report with all violation data |
| ⚡ **Performance Optimization** | Low-End Hardware Support | YOLO frame-skipping, inference resizing, and display throttling |

---

## 🛠️ Tech Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core language | 3.11+ |
| **OpenCV** | Video capture, frame processing, annotations | 5.0 |
| **MediaPipe Face Detection** | Real-time face localization | 0.10.35 |
| **MediaPipe Face Mesh** | 468-point facial landmark extraction for gaze analysis | 0.10.35 |
| **YOLOv8n** | Object detection (cell phone identification) | 8.4 |
| **Streamlit** | Interactive web dashboard | 1.59 |
| **Pandas** | Data structuring and CSV report generation | 3.0 |
| **NumPy** | Numerical operations and array manipulation | 2.4 |

---

## 🏗️ System Architecture

The processing pipeline follows a sequential flow — each frame passes through all detection stages before the results are aggregated by the warning system and displayed on the dashboard.

```
┌─────────────────────────────────────────────────────────────┐
│                     WEBCAM CAPTURE                          │
│                   (OpenCV VideoCapture)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ BGR Frame (640×480)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FRAME PREPROCESSING                        │
│          BGR → RGB conversion (once per frame)              │
│          Resize to 320px for inference                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ RGB Frame (320×240)
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌──────────────────┐ ┌──────────┐ ┌──────────────────┐
│  FACE DETECTOR   │ │   GAZE   │ │  PHONE DETECTOR  │
│   (MediaPipe     │ │ DETECTOR │ │   (YOLOv8n)      │
│ Face Detection)  │ │(FaceMesh)│ │ Every 5th frame   │
│                  │ │          │ │ imgsz=320         │
│ • Face count     │ │ • Yaw    │ │                   │
│ • Bounding boxes │ │ • Pitch  │ │ • Phone bbox      │
│ • Confidence     │ │ • Gaze   │ │ • Confidence      │
└────────┬─────────┘ └────┬─────┘ └────────┬──────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    WARNING SYSTEM                           │
│                                                             │
│  Violation Weights:                                         │
│  • Phone Usage     ×50  (Critical)                          │
│  • Multiple Faces  ×30  (High)                              │
│  • No Face         ×5   (Medium)                            │
│  • Looking Away    ×2   (Minor)                             │
│                                                             │
│  Risk Levels: Low (<50) │ Medium (50-149) │ High (≥150)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│   VIDEO OVERLAY      │  │       STREAMLIT DASHBOARD         │
│                      │  │                                   │
│ • Face bounding box  │  │ • Risk Level metric               │
│ • Phone bounding box │  │ • Risk Score                      │
│ • Gaze landmarks     │  │ • FPS counter                     │
│ • Risk / FPS text    │  │ • Violation counter table          │
└──────────────────────┘  │ • CSV Report download             │
                          └───────────────────────────────────┘
```

---

## 📁 Project Structure

```
AI_Interview_Proctor/
│
├── src/                          # Core detection modules
│   ├── __init__.py               # Package initializer
│   ├── webcam.py                 # Thread-safe webcam capture with OpenCV
│   ├── face_detector.py          # MediaPipe face detection & bounding boxes
│   ├── gaze_detector.py          # MediaPipe Face Mesh gaze & head pose analysis
│   ├── phone_detector.py         # YOLOv8n phone detection with frame-skipping
│   ├── warning_system.py         # Violation aggregation & risk score calculation
│   └── report_generator.py       # Pandas-based CSV report generation
│
├── models/                       # Model weights directory
├── reports/                      # Generated CSV session reports
│
├── app.py                        # Main Streamlit dashboard (entry point)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11** or higher
- **pip** package manager
- A working **webcam** (built-in or USB)
- **Windows 10/11**, macOS, or Linux

### Step 1 — Clone the Repository

```bash
git clone https://github.com/adarshvij/AI_Interview_Proctor.git
cd AI_Interview_Proctor
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

**Activate the environment:**

| OS | Command |
|----|---------|
| Windows (PowerShell) | `.\venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.\venv\Scripts\activate.bat` |
| macOS / Linux | `source venv/bin/activate` |

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8n model weights (`yolov8n.pt`) are downloaded automatically on first run by the Ultralytics library. No manual download is required.

### Step 4 — Verify Installation

```bash
python -c "import cv2, mediapipe, ultralytics, streamlit, pandas; print('All dependencies installed successfully')"
```

---

## 💻 Usage

### Start the Application

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

### Dashboard Controls

| Control | Location | Description |
|---------|----------|-------------|
| **▶️ Start Proctoring** | Sidebar | Begins webcam monitoring and violation tracking |
| **⏹️ Stop Proctoring** | Sidebar | Stops the monitoring session |
| **Face Detection Confidence** | Sidebar slider | Adjusts MediaPipe sensitivity (default: 0.5) |
| **Phone Detection Confidence** | Sidebar slider | Adjusts YOLO threshold (default: 0.25) |
| **Candidate Name / ID** | Sidebar inputs | Metadata included in generated reports |
| **📄 Generate Report** | Dashboard panel | Exports session data as a downloadable CSV |

### Typical Workflow

1. Enter the candidate's name and ID in the sidebar.
2. Adjust detection confidence thresholds if needed.
3. Click **▶️ Start Proctoring** to begin monitoring.
4. The live feed displays face boxes, gaze landmarks, phone detections, risk level, and FPS.
5. The dashboard panel shows real-time violation counters and risk scores.
6. When finished, click **⏹️ Stop Proctoring**.
7. Click **📄 Generate Report** to download the session summary as CSV.

---

## ⚡ Performance Optimization

The system was engineered to run smoothly on standard hardware without a dedicated GPU.

### Target Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7-8565U (4 cores / 8 threads) |
| GPU | Intel UHD Graphics 620 (integrated) |
| RAM | 8 GB DDR4 |

### Optimizations Applied

| # | Optimization | Technique | Impact |
|---|-------------|-----------|--------|
| 1 | **YOLO Frame Skipping** | Run YOLOv8 inference every 5th frame; cache results for intermediate frames | ~40-60ms saved per skipped frame |
| 2 | **Reduced Inference Resolution** | Resize frames to 320px width before passing to MediaPipe; YOLO uses `imgsz=320` | ~50% faster inference |
| 3 | **Single RGB Conversion** | Convert BGR→RGB once per frame and share across face and gaze detectors | ~2ms saved per frame |
| 4 | **In-Place Drawing** | All annotation methods draw directly on the frame without creating copies | ~3-5ms saved + reduced memory |
| 5 | **Streamlit Display Throttling** | Cap dashboard image updates to ~15 FPS to avoid PNG encoding bottleneck | ~15-30ms saved on skipped frames |
| 6 | **Adaptive Sleep** | Replace fixed `sleep(0.03)` with dynamic sleep based on actual processing time | Maximizes usable FPS |
| 7 | **Model Caching** | Cache detector instances in Streamlit session state across reruns | Eliminates multi-second model reload |
| 8 | **FPS Monitoring** | Rolling 30-frame average displayed on overlay and dashboard | Real-time performance visibility |

### Performance Results

| Metric | Before Optimization | After Optimization |
|--------|--------------------|--------------------|
| Frame Processing Time | ~105-145ms | ~25-45ms |
| Effective FPS | 7-10 FPS | 22-35 FPS |
| Improvement | — | **~3× faster** |

---

## 🧩 Challenges & Solutions

### 1. Dashboard Freezing During Monitoring

**Challenge:** Streamlit's synchronous execution model meant the `while` loop blocked the main thread. Dashboard widgets rendered once before the loop and never updated, causing the video overlay and dashboard to show different risk levels.

**Solution:** Used `st.empty()` placeholders for all dashboard widgets and updated them inside the processing loop. This makes the dashboard live without breaking Streamlit's architecture.

---

### 2. YOLO Consuming All CPU Budget

**Challenge:** YOLOv8n inference takes ~50-80ms per frame on an integrated GPU. Running it on every frame at 640×480 left no CPU budget for other detectors, resulting in single-digit FPS.

**Solution:** Implemented frame-skipping (YOLO runs every 5th frame) with result caching. Phone presence changes on the order of seconds, so a 1/5th-second delay is imperceptible. Combined with `imgsz=320`, this reduced YOLO's amortized cost to ~10ms per frame.

---

### 3. Counter Values Not Updating in UI

**Challenge:** The warning system returned a reference to its internal counters dictionary. Since the dictionary object identity never changed, Streamlit's diffing engine couldn't detect updates.

**Solution:** Returned `self.counters.copy()` to create independent snapshots. Each update now produces a new dictionary object that Streamlit correctly recognizes as changed state.

---

### 4. Model Reloading on Every Interaction

**Challenge:** MediaPipe and YOLO models were instantiated inside the Streamlit rerun block. Every button click or slider adjustment destroyed and rebuilt all models, causing multi-second freezes.

**Solution:** Cached model instances in `st.session_state` so they persist across reruns. Models are only re-created when the user explicitly starts a new session.

---

### 5. Duplicate Computation Across Detectors

**Challenge:** Both the face detector and gaze detector independently converted frames from BGR to RGB — the same ~2ms operation running twice per frame with identical results.

**Solution:** Centralized the BGR→RGB conversion in the main pipeline and added `detect_faces_rgb()` and `detect_gaze_rgb()` methods that accept pre-converted frames. The original BGR methods are preserved for backward compatibility.

---

## 🔮 Future Improvements

| Priority | Improvement | Description |
|----------|-------------|-------------|
| 🔴 High | **Audio Monitoring** | Detect background voices or suspicious audio patterns using speech recognition |
| 🔴 High | **Tab/Screen Switching Detection** | Monitor browser focus events to flag when candidates switch tabs |
| 🟡 Medium | **Identity Verification** | Face recognition to verify the candidate matches their registered photo |
| 🟡 Medium | **Timestamped Event Log** | Record each violation with frame timestamps for detailed post-session review |
| 🟡 Medium | **Multi-Camera Support** | Enable side-camera or environment-camera feeds for broader coverage |
| 🟢 Low | **GPU Acceleration** | CUDA/TensorRT inference for systems with NVIDIA GPUs |
| 🟢 Low | **Cloud Deployment** | Dockerized deployment on AWS/GCP with WebRTC for remote proctoring |
| 🟢 Low | **Admin Panel** | A separate dashboard for administrators to monitor multiple candidates simultaneously |

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 👤 Author

**Adarsh Vij**

- GitHub: [adarshvij](https://github.com/adarshvij)
- LinkedIn: *Coming soon*

---

<div align="center">

⭐ **If you found this project useful, consider giving it a star!** ⭐

Built with ❤️ using Python, OpenCV, MediaPipe, YOLOv8, and Streamlit

</div>
