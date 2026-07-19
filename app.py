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
    gaze_tolerance = st.sidebar.slider("Gaze Deviation Tolerance", 1, 10, 5, 1)
    
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
        else:
            # Active monitoring simulation/loop
            webcam = WebcamStream()
            face_detector = FaceDetector(min_detection_confidence=face_conf)
            phone_detector = PhoneDetector(min_confidence=phone_conf)
            gaze_detector = GazeDetector(tolerance=gaze_tolerance)
            
            # Start stream
            if webcam.start():
                try:
                    while st.session_state.proctoring_active:
                        frame = webcam.read()
                        if frame is None:
                            st.warning("No frame received from webcam. Reconnecting...")
                            time.sleep(0.5)
                            continue
                        
                        # Pipeline operations (Placeholders)
                        # 1. Face detection
                        faces, face_count = face_detector.detect_faces(frame)
                        
                        # 2. Gaze tracking
                        gaze_info = gaze_detector.track_gaze(frame, faces)
                        
                        # 3. Phone detection
                        phones = phone_detector.detect_phones(frame)
                        
                        # 4. Feed warning system & get status
                        status = st.session_state.warning_system.update(
                            face_count=face_count,
                            is_looking_away=gaze_info.get("looking_away", False),
                            phone_detected=len(phones) > 0,
                            frame=frame
                        )
                        
                        # Annotate frame for display (Mock drawing)
                        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Draw placeholders or detection borders on annotated_frame
                        if face_count > 0:
                            cv2.putText(
                                annotated_frame,
                                f"Faces: {face_count}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                        
                        # Render video stream in Streamlit
                        video_placeholder.image(annotated_frame, channels="RGB")
                        
                        # Prevent high CPU utilization
                        time.sleep(0.03)
                finally:
                    webcam.release()
            else:
                st.error("Failed to initialize webcam. Please check camera connection.")

    with side_col:
        st.subheader("Live Proctoring Status")
        
        # Threat assessment / Risk score metric
        risk_score = st.session_state.warning_system.get_risk_score()
        st.metric(
            label="Risk Score",
            value=f"{risk_score}%",
            delta="-Good" if risk_score < 30 else "⚠️ Moderate Risk" if risk_score < 70 else "🚨 High Risk",
            delta_color="inverse"
        )
        
        # Display progress indicators
        st.progress(risk_score / 100.0)
        
        # Real-time incident logs
        st.subheader("Recent Violation Events")
        violation_logs = st.session_state.warning_system.get_logs()
        if violation_logs:
            df = pd.DataFrame(violation_logs)
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("No violations recorded yet.")
            
        # Report generator trigger
        st.markdown("---")
        st.subheader("Reports")
        if st.button("📄 Generate Proctoring Report", use_container_width=True):
            with st.spinner("Compiling reports, plots and logs..."):
                generator = ReportGenerator(
                    candidate_name=candidate_name,
                    candidate_id=candidate_id,
                    logs=violation_logs
                )
                report_path = generator.generate_report()
                st.success(f"Report generated successfully at: {report_path}")
                
                # Mock Download button
                st.download_button(
                    label="Download CSV Log",
                    data=pd.DataFrame(violation_logs).to_csv(index=False),
                    file_name=f"{candidate_id}_proctor_log.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
