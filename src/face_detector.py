"""
Face Detector Module

Utilizes MediaPipe or OpenCV to perform real-time face detection,
verifying candidate presence and flagging multiple face infractions.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import mediapipe as mp
import cv2

class FaceDetector:
    """
    Handles face tracking, presence verification, and count assessment.
    Integrates MediaPipe Face Detection to report candidate status.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initializes the FaceDetector configuration.

        Args:
            min_detection_confidence (float): Minimum confidence threshold for face detection.
        """
        self.min_detection_confidence = min_detection_confidence
        # Placeholder for MediaPipe face detection instance
        self.mp_face_detection = None
        self.face_detection_model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Loads the MediaPipe Face Detection model.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_model = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence
        )

    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], int]:
        """
        Processes a video frame to detect faces and compute bounding boxes.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.

        Returns:
            Tuple[List[Dict[str, Any]], int]: A list of detected faces metadata (bounding box, landmarks, etc.)
                                              and the total count of faces detected.
        """
        # 1. Convert the frame from BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._process_rgb(rgb_frame, frame.shape)

    def detect_faces_rgb(self, rgb_frame: np.ndarray, original_shape: tuple = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        [Performance P2] Processes an already-converted RGB frame.
        Avoids redundant BGR->RGB conversion when the caller has already done it.

        Args:
            rgb_frame (np.ndarray): The RGB image frame.
            original_shape (tuple, optional): (h, w, c) of the original full-res frame.
                If provided and different from rgb_frame.shape, bounding boxes are
                scaled back to original coordinates. This lets us run inference on
                a smaller frame while drawing boxes on the full-res display frame.

        Returns:
            Tuple[List[Dict[str, Any]], int]: Detected faces and count.
        """
        # Use original shape for coordinate mapping if provided
        ref_shape = original_shape if original_shape else rgb_frame.shape
        return self._process_rgb(rgb_frame, ref_shape)

    def _process_rgb(self, rgb_frame: np.ndarray, ref_shape: tuple) -> Tuple[List[Dict[str, Any]], int]:
        """
        Internal: runs MediaPipe on an RGB frame and maps coordinates to ref_shape.

        Args:
            rgb_frame (np.ndarray): RGB image for MediaPipe.
            ref_shape (tuple): (h, w, c) used for coordinate de-normalization.
        """
        # 2. Process the frame using the initialized MediaPipe model
        results = self.face_detection_model.process(rgb_frame)
        
        faces_detected = []
        
        # 3. If faces are found, extract the bounding box and confidence score
        if results.detections:
            # Use ref_shape dimensions to convert normalized coordinates to pixels
            ih, iw = ref_shape[0], ref_shape[1]
            
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                # Extract coordinates, convert to pixels, and grab confidence score
                face_data = {
                    "x": int(bboxC.xmin * iw),
                    "y": int(bboxC.ymin * ih),
                    "width": int(bboxC.width * iw),
                    "height": int(bboxC.height * ih),
                    "confidence": float(detection.score[0])
                }
                faces_detected.append(face_data)
                
        # 4. Return the list of faces and the total count
        return faces_detected, len(faces_detected)

    def check_violations(self, face_count: int) -> Tuple[bool, str]:
        """
        Analyzes the face count to check for candidate presence anomalies.

        Args:
            face_count (int): Number of detected faces.

        Returns:
            Tuple[bool, str]: A tuple containing violation status (True/False) and violation description.
        """
        if face_count == 0:
            return True, "No face detected"
        elif face_count > 1:
            return True, f"Multiple faces detected: {face_count}"
        return False, "Normal"

    def draw_faces(self, frame: np.ndarray, faces_detected: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws bounding boxes and confidence scores around detected faces.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.
            faces_detected (List[Dict[str, Any]]): List of faces containing coordinate data.

        Returns:
            np.ndarray: The modified image frame with drawn annotations.
        """
        # [Performance P3] Draw directly on the frame passed in (no copy).
        # The caller is responsible for making a copy if the original is needed.
        # This saves ~1-2ms per frame by avoiding a full-resolution array copy.
        annotated_frame = frame

        for face in faces_detected:
            x, y, w, h = face["x"], face["y"], face["width"], face["height"]
            conf = face["confidence"]

            # Draw a green rectangle around the face (BGR format: 0, 255, 0)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the confidence score slightly above the bounding box
            text = f"{conf * 100:.1f}%"
            cv2.putText(annotated_frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame
