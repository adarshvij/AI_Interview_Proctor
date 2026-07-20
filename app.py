"""
AI Interview Proctor - Main Streamlit Dashboard

This is the main entrypoint of the application. It coordinates video streaming,
runs the frame processing pipeline through various detector classes, keeps track
of warnings/risk scores, and manages UI rendering.
"""

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
from src.webcam import WebcamStream
from src.face_detector import FaceDetector
from src.phone_detector import PhoneDetector
from src.gaze_detector import GazeDetector
from src.warning_system import WarningSystem
from src.report_generator import ReportGenerator

# Set up page configurations
st.set_page_config(
    page_title="AI Interview Proctor Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables
if "proctoring_active" not in st.session_state:
    st.session_state.proctoring_active = False

if "warning_system" not in st.session_state:
    st.session_state.warning_system = WarningSystem()

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result = {"counters": {"multiple_faces": 0, "phone_usage": 0, "looking_away": 0, "no_face": 0}, "risk_level": "Low"}

# [Bug fix B3] Cache detector instances in session_state so they survive
# Streamlit reruns (slider changes, button clicks). Without this, every rerun
# destroys and rebuilds all MediaPipe/YOLO models — causing multi-second freezes
# and the inference_feedback_manager warnings in the logs.
if "face_detector" not in st.session_state:
    st.session_state.face_detector = None
if "phone_detector" not in st.session_state:
    st.session_state.phone_detector = None
if "gaze_detector" not in st.session_state:
    st.session_state.gaze_detector = None

# --- Performance constants ---
# [Performance P4] Width to resize inference frames to.
# MediaPipe works well at 320px; this halves pixel count vs 640px.
INFERENCE_WIDTH = 320

# [Performance P5] Minimum interval between Streamlit image updates (seconds).
# Streamlit's image() encodes to PNG, base64-encodes, and sends over WebSocket.
# At ~20ms per call, capping to ~15 FPS display prevents this from dominating.
MIN_DISPLAY_INTERVAL = 0.066  # ~15 FPS display cap

# [Performance P6] Target frame time for adaptive sleep (seconds).
# Instead of a fixed sleep(0.03), we sleep only the remaining time to hit the
# target, so fast frames don't waste time and slow frames aren't penalized.
TARGET_FRAME_TIME = 0.033  # ~30 FPS target


def main():
    st.title("🛡️ AI Interview Proctoring System")
    st.write(
        "Real-time AI-based monitoring dashboard for detecting candidate malpractices "
        "during remote assessments."
    )

    # Sidebar settings
    st.sidebar.header("Proctoring Configurations")
    
    # Threshold controls
    face_conf = st.sidebar.slider("Min Face Detection Confidence", 0.0, 1.0, 0.5, 0.05)
    phone_conf = st.sidebar.slider("Min Phone Detection Confidence", 0.0, 1.0, 0.25, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Candidate Information")
    candidate_name = st.sidebar.text_input("Name", "John Doe")
    candidate_id = st.sidebar.text_input("Candidate ID", "CAN-2026-001")
    
    # Dashboard Control Buttons
    st.sidebar.subheader("Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start Proctoring", use_container_width=True):
            st.session_state.proctoring_active = True
            st.session_state.warning_system = WarningSystem()  # Reset system
            st.session_state.start_time = time.time()
            # [Bug fix B3] Reset cached detectors so they pick up new confidence
            # thresholds from the sliders on the next initialization.
            st.session_state.face_detector = None
            st.session_state.phone_detector = None
            st.session_state.gaze_detector = None
    with col2:
        if st.button("⏹️ Stop Proctoring", use_container_width=True):
            st.session_state.proctoring_active = False

    # Main dashboard layout: 2 columns
    main_col, side_col = st.columns([2, 1])

    with main_col:
        st.subheader("Live Feed & Detections")
        video_placeholder = st.empty()
        
        # Display instructions when not active
        if not st.session_state.proctoring_active:
            video_placeholder.info(
                "Proctoring is currently idle. Click 'Start Proctoring' in the sidebar to begin."
            )

    # [Bug fix B1] Create EMPTY placeholders for all dashboard widgets.
    # These will be updated INSIDE the processing loop so the dashboard stays
    # live. Previously, these widgets were rendered once before the loop and
    # never updated — causing the overlay/dashboard mismatch.
    with side_col:
        st.subheader("Live Proctoring Status")
        risk_placeholder = st.empty()
        fps_placeholder = st.empty()
        st.subheader("Violation Counters")
        counter_placeholder = st.empty()
        
        # Report generator trigger (outside the loop — always visible)
        st.markdown("---")
        st.subheader("Reports")
        if st.button("📄 Generate Proctoring Report", use_container_width=True):
            with st.spinner("Compiling report..."):
                # [Bug fix B5] Read live data from session_state, not stale vars.
                # Previously, `counters` and `risk_level` were captured before the
                # loop started, so reports always contained initial zero values.
                live_result = st.session_state.latest_result
                live_counters = live_result["counters"]
                live_risk = live_result["risk_level"]
                
                # Calculate session duration
                duration = 0.0
                if st.session_state.start_time is not None:
                    duration = time.time() - st.session_state.start_time
                
                generator = ReportGenerator()
                df = generator.compile_data(
                    duration_seconds=duration,
                    counters=live_counters,
                    risk_level=live_risk
                )
                filepath = generator.export_to_csv(df)
                st.success(f"Report saved to: {filepath}")
                
                # Download button
                st.download_button(
                    label="Download CSV Report",
                    data=df.to_csv(index=False),
                    file_name=f"{candidate_id}_proctor_report.csv",
                    mime="text/csv"
                )

    # === Active monitoring loop ===
    if st.session_state.proctoring_active:
        # [Bug fix B3] Initialize detectors only if not already cached.
        # This prevents re-creating MediaPipe/YOLO models on every Streamlit
        # rerun (which happens on any button click or slider change).
        if st.session_state.face_detector is None:
            st.session_state.face_detector = FaceDetector(min_detection_confidence=face_conf)
        if st.session_state.phone_detector is None:
            st.session_state.phone_detector = PhoneDetector(min_confidence=phone_conf)
        if st.session_state.gaze_detector is None:
            st.session_state.gaze_detector = GazeDetector()
        
        face_detector = st.session_state.face_detector
        phone_detector = st.session_state.phone_detector
        gaze_detector = st.session_state.gaze_detector

        webcam = WebcamStream()
        
        # [Performance P8] FPS measurement state
        fps = 0.0
        frame_times = []  # Rolling window for FPS calculation

        # Track last display update time for throttling
        last_display_time = 0.0

        if webcam.start():
            try:
                while st.session_state.proctoring_active:
                    # [Performance P6] Record frame start for adaptive sleep
                    frame_start = time.time()

                    frame = webcam.read()
                    if frame is None:
                        st.warning("No frame received from webcam. Reconnecting...")
                        time.sleep(0.5)
                        continue
                    
                    # === Optimized Pipeline ===
                    
                    # [Performance P2] Convert BGR→RGB ONCE per frame.
                    # Both FaceDetector and GazeDetector need RGB. Previously each
                    # did its own cvtColor — wasting ~2ms per frame on duplicate work.
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # [Performance P4] Resize for inference.
                    # MediaPipe works well at 320px wide. This reduces pixel count by
                    # ~75% (320×240 vs 640×480), speeding up both face and gaze detection.
                    h, w = frame.shape[:2]
                    scale = INFERENCE_WIDTH / w
                    small_h = int(h * scale)
                    small_rgb = cv2.resize(rgb_frame, (INFERENCE_WIDTH, small_h))
                    
                    # 1. Face detection — uses small RGB frame, maps coords to original
                    faces, face_count = face_detector.detect_faces_rgb(
                        small_rgb, original_shape=frame.shape
                    )
                    
                    # 2. Gaze tracking — uses small RGB frame (landmark coords are normalized)
                    landmarks, gaze_data = gaze_detector.detect_gaze_rgb(small_rgb)
                    
                    # 3. Phone detection — frame-skip handled internally by PhoneDetector.
                    #    YOLO runs every 5th frame; cached result returned otherwise.
                    phones, phone_count = phone_detector.detect_phones(frame)
                    
                    # 4. Feed warning system & get status
                    warning_result = st.session_state.warning_system.update(
                        face_count,
                        gaze_data.get("looking_away", False),
                        phone_count > 0
                    )
                    st.session_state.latest_result = warning_result
                    
                    # 5. Annotate frame — all draw methods now work IN-PLACE (no copies).
                    #    We make ONE copy here for the display frame, then draw on it.
                    annotated_frame = frame.copy()
                    
                    # [Bug fix B4] Draw ALL detections, not just faces.
                    # Previously, only draw_faces() was called. Phone bounding boxes
                    # and gaze landmarks were never rendered on the video.
                    face_detector.draw_faces(annotated_frame, faces)
                    phone_detector.draw_phones(annotated_frame, phones)
                    if len(landmarks) > 0:
                        gaze_detector.draw_landmarks(annotated_frame, landmarks)
                    
                    # Draw status overlay
                    risk_level = warning_result["risk_level"]
                    color = (0, 255, 0) if risk_level == "Low" else (0, 165, 255) if risk_level == "Medium" else (0, 0, 255)
                    cv2.putText(annotated_frame, f"Faces: {face_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(annotated_frame, f"Gaze: {gaze_data.get('direction', 'N/A')}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(annotated_frame, f"Phones: {phone_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(annotated_frame, f"Risk: {risk_level}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # [Performance P8] Display FPS on overlay
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # [Performance P5] Throttle Streamlit display updates.
                    # Streamlit's image() is expensive (~15-30ms per call due to PNG
                    # encoding + base64 + WebSocket). We cap display to ~15 FPS.
                    now = time.time()
                    if now - last_display_time >= MIN_DISPLAY_INTERVAL:
                        # Convert BGR to RGB for Streamlit display
                        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(display_frame, channels="RGB")
                        
                        # [Bug fix B1] Update dashboard placeholders INSIDE the loop.
                        # This makes the dashboard live — risk level, score, counters,
                        # and FPS all update in real-time alongside the video overlay.
                        counters = warning_result["counters"]
                        risk_score = st.session_state.warning_system.total_risk_score
                        
                        risk_placeholder.metric(
                            label="Risk Level",
                            value=risk_level,
                            delta=f"Score: {risk_score}",
                            delta_color="inverse"
                        )
                        
                        fps_placeholder.metric(label="FPS", value=f"{fps:.1f}")
                        
                        counter_df = pd.DataFrame({
                            "Violation Type": ["Multiple Faces", "Phone Usage", "Looking Away", "No Face"],
                            "Count": [
                                counters["multiple_faces"],
                                counters["phone_usage"],
                                counters["looking_away"],
                                counters["no_face"]
                            ]
                        })
                        counter_placeholder.dataframe(counter_df, use_container_width=True, hide_index=True)
                        
                        last_display_time = now
                    
                    # [Performance P8] Calculate FPS using a rolling window.
                    # We average the last 30 frame times for a stable reading.
                    frame_end = time.time()
                    frame_times.append(frame_end - frame_start)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    if frame_times:
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    
                    # [Performance P6] Adaptive sleep: only sleep the remaining time
                    # to reach the target frame rate. If processing already took longer
                    # than the target, don't sleep at all. This replaces the fixed
                    # time.sleep(0.03) which added delay regardless of processing time.
                    elapsed = frame_end - frame_start
                    sleep_time = TARGET_FRAME_TIME - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                webcam.release()
        else:
            st.error("Failed to initialize webcam. Please check camera connection.")

if __name__ == "__main__":
    main()
