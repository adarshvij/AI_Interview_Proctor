# AI Interview Proctor

An automated, real-time AI-based interview proctoring system designed to ensure exam and interview integrity by detecting cheating behaviors using computer vision and deep learning techniques.

## Problem Statement

With the rise of online recruitment and remote examinations, maintaining the academic and professional integrity of assessments has become a significant challenge. Manual remote proctoring is resource-intensive and prone to human oversight. 

**AI Interview Proctor** solves this problem by providing automated, multi-modal surveillance during online assessments. It monitors the candidate via webcam in real-time, detecting anomalies such as looking away, absence from the screen, presence of multiple individuals, and unauthorized mobile phone usage, compiling all logs into a comprehensive risk report.

## Features

- **Real-Time Webcam Streaming**: Multi-threaded, low-latency video frame acquisition.
- **Face & Multiple Face Detection**: Monitors if the candidate is present and ensures no second person is in the frame.
- **Mobile Phone Detection**: Employs a pre-trained lightweight YOLOv8 object detection model to detect unauthorized phone usage.
- **Gaze & Head Pose Estimation**: Tracks eye gaze direction and head orientation to flag when a candidate repeatedly looks away from the screen.
- **Warning & Risk Score Management**: Tracks violations dynamically and calculates a cumulative risk score.
- **Automated Report Generation**: Compiles assessment logs, charts, and statistics into structured reports.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Web-based user interface and proctoring dashboard.
- **OpenCV**: Image pre-processing, video capturing, and frame operations.
- **MediaPipe**: Real-time face detection, face landmark extraction, and head-pose computation.
- **YOLOv8 (Ultralytics)**: Object detection for unauthorized device/phone identification.
- **NumPy & Pandas**: Data manipulation and numerical operations on tracking timelines.
- **Matplotlib**: Statistical chart generation for the final reports.

## Folder Structure

```text
AI_Interview_Proctor/
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Python dependency specifications
├── README.md               # Project documentation
├── .gitignore              # Files and folders ignored by git
├── dataset/                # Dataset directory for model calibration/training
│   ├── normal/             # Frames representing honest test-taking behaviour
│   ├── phone_usage/        # Frames representing unauthorized phone usage
│   ├── multiple_faces/     # Frames with more than one person visible
│   ├── looking_away/       # Frames where the test taker is distracted
│   └── no_face/            # Frames with no candidate present
├── models/                 # Pre-trained and custom weights (e.g. YOLOv8)
├── reports/                # Generated PDF/CSV/HTML candidate reports
├── screenshots/            # Saved violation screenshots for audit trails
└── src/                    # Source files containing modular logic
    ├── webcam.py           # Thread-safe webcam stream controller
    ├── face_detector.py    # MediaPipe face presence and count analyser
    ├── phone_detector.py   # YOLOv8 object detection integration
    ├── gaze_detector.py    # Face landmark & eye gaze vector estimator
    ├── warning_system.py   # Warning triggers, logs and risk calculations
    └── report_generator.py # Visual report assembler
```

## Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/AI_Interview_Proctor.git
   cd AI_Interview_Proctor
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Future Scope

- **Audio Proctoring**: Integration of microphone audio monitoring to detect background whispers, speech recognition for prohibited phrases, or voice activity detection (VAD).
- **Tab Switching Detection**: Browser-level extension or frontend tracking to monitor if a candidate switches tabs or windows.
- **Low Light Optimization**: Enhance facial landmark extraction under suboptimal lighting conditions using image enhancement algorithms (e.g., CLAHE).
- **Custom Model Fine-tuning**: Fine-tune YOLOv8 on specific classroom/office object datasets (detecting books, second monitors, etc.).
- **Cloud Database Integration**: Securely upload violation logs and candidate scores to a central cloud database for HR/Administrator access.
