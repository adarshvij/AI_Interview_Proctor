"""
Gaze & Head Posture Detector Module

Utilizes MediaPipe Face Mesh to extract dense facial landmarks.
Evaluates head pose (yaw, pitch, roll) and eye gaze direction to identify
when a candidate is looking away from the screen.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import mediapipe as mp
import cv2

class GazeDetector:
    """
    Computes facial coordinate transformations to estimate gaze and 3D head orientation.
    Integrates MediaPipe Face Mesh.
    """

    def __init__(self):
        """
        Initializes the GazeDetector settings.
        """
        self.mp_face_mesh = None
        self.face_mesh_model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Loads the MediaPipe Face Mesh model into memory.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_gaze(self, frame: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Processes facial landmarks from the frame to extract gaze direction and head orientation.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.

        Returns:
            Tuple: A list of the raw facial landmarks, and a dictionary of the calculated gaze data.
        """
        # 1. Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._process_rgb(rgb_frame)

    def detect_gaze_rgb(self, rgb_frame: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
        """
        [Performance P2] Processes an already-converted RGB frame.
        Avoids redundant BGR->RGB conversion when the caller has already done it.

        Args:
            rgb_frame (np.ndarray): The RGB image frame.

        Returns:
            Tuple: A list of raw facial landmarks, and a dictionary of calculated gaze data.
        """
        return self._process_rgb(rgb_frame)

    def _process_rgb(self, rgb_frame: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Internal: runs Face Mesh on an RGB frame and computes gaze direction.
        """
        # 2. Run the Face Mesh model
        results = self.face_mesh_model.process(rgb_frame)
        
        # Default state
        gaze_data = {
            "looking_away": False,
            "direction": "Center"
        }
        raw_landmarks = []
        
        if results.multi_face_landmarks:
            # We only evaluate the primary face [0]
            face_landmarks = results.multi_face_landmarks[0]
            raw_landmarks = face_landmarks.landmark
            
            # 3. Extract 2D coordinates for Nose (1), Left Eye (33), Right Eye (263)
            nose = raw_landmarks[1]
            left_eye = raw_landmarks[33]
            right_eye = raw_landmarks[263]
            
            # 4. Calculate Horizontal (Yaw) Ratio
            dist_left = nose.x - left_eye.x
            dist_right = right_eye.x - nose.x
            
            # Prevent division by zero just in case
            if dist_right == 0: dist_right = 0.001
            yaw_ratio = dist_left / dist_right
            
            # 5. Calculate Vertical (Pitch) Ratio
            # We normalize the vertical distance by the width of the eyes
            eye_width = right_eye.x - left_eye.x
            if eye_width == 0: eye_width = 0.001
            pitch_ratio = (nose.y - left_eye.y) / eye_width
            
            # 6. Apply Geometric Thresholds
            if yaw_ratio < 0.7:
                gaze_data["direction"] = "Looking Left"
                gaze_data["looking_away"] = True
            elif yaw_ratio > 1.4:
                gaze_data["direction"] = "Looking Right"
                gaze_data["looking_away"] = True
            elif pitch_ratio > 0.8:
                gaze_data["direction"] = "Looking Down"
                gaze_data["looking_away"] = True
            else:
                gaze_data["direction"] = "Center"
                gaze_data["looking_away"] = False
                
        return raw_landmarks, gaze_data

    def check_violations(self, gaze_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Analyzes the gaze data to check if the user is looking away.

        Args:
            gaze_data (Dict[str, Any]): The computed gaze angles and direction.

        Returns:
            Tuple[bool, str]: A tuple containing violation status (True/False) and violation description.
        """
        if gaze_data.get("looking_away", False):
            return True, gaze_data.get("direction", "Looking Away")
        return False, "Center"

    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Any]) -> np.ndarray:
        """
        Draws the dense facial mesh or specific gaze tracking points on the face.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.
            landmarks (List[Any]): List of facial landmark points from MediaPipe.

        Returns:
            np.ndarray: The modified image frame with drawn annotations.
        """
        # [Performance P3] Draw directly on the frame passed in (no copy).
        # The caller is responsible for making a copy if the original is needed.
        # This saves ~1-2ms per frame by avoiding a full-resolution array copy.
        annotated_frame = frame

        if len(landmarks) == 0:
            return annotated_frame

        ih, iw, _ = frame.shape

        # Draw the 3 key landmarks used for gaze calculation
        # Nose (1), Left Eye (33), Right Eye (263)
        key_indices = [1, 33, 263]
        labels = ["Nose", "L-Eye", "R-Eye"]

        for idx, label in zip(key_indices, labels):
            lm = landmarks[idx]
            cx, cy = int(lm.x * iw), int(lm.y * ih)

            # Draw a cyan circle at each key landmark
            cv2.circle(annotated_frame, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(annotated_frame, label, (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        return annotated_frame
