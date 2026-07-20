"""
Phone Detector Module

Utilizes YOLOv8n to perform real-time object detection,
specifically looking for cell phones to flag unauthorized device usage.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO

class PhoneDetector:
    """
    Handles cell phone detection and presence verification.
    Integrates Ultralytics YOLOv8n to report if the candidate is using a phone.
    """

    def __init__(self, min_confidence: float = 0.3, frame_skip: int = 5):
        """
        Initializes the PhoneDetector configuration.

        Args:
            min_confidence (float): Minimum confidence threshold for phone detection.
            frame_skip (int): Run YOLO only every N-th frame to reduce CPU load.
        """
        self.min_confidence = min_confidence
        self.model = None
        # [Performance P1] Frame-skip state: YOLO is the heaviest detector
        # (~50-80ms on CPU). Running it every frame wastes most of the CPU budget.
        # We run it every N-th frame and cache the result for intermediate frames.
        self.frame_skip = frame_skip
        self._frame_counter = 0
        self._cached_phones: list = []
        self._cached_count: int = 0
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Loads the YOLOv8n Object Detection model into memory.
        """
        self.model = YOLO("yolov8n.pt")

    def detect_phones(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], int]:
        """
        Processes a video frame to detect cell phones (COCO Dataset class 67).
        [Performance P1] Uses frame-skipping: only runs YOLO every N-th frame.
        Returns cached results on intermediate frames.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.

        Returns:
            Tuple[List[Dict[str, Any]], int]: A list of detected phone metadata (bounding box, confidence)
                                              and the total count of phones detected.
        """
        # [Performance P1] Skip YOLO on intermediate frames — reuse cached result.
        # Phone presence changes slowly (seconds, not frames), so stale-by-1/5th
        # of a second is acceptable and saves ~40-60ms of CPU per skipped frame.
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            return self._cached_phones, self._cached_count

        # [Performance P4] Use imgsz=320 so YOLO internally resizes to 320×320.
        # This halves inference time vs the default 640×640 with minimal accuracy
        # loss for detecting a large object like a phone held near the camera.
        results = self.model(frame, classes=[67], conf=self.min_confidence,
                             verbose=False, imgsz=320)
        
        phones_detected = []
        
        # Iterate through the results and extract bounding boxes
        for r in results:
            for box in r.boxes:
                # YOLO returns xyxy format (top-left and bottom-right points)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate width and height for consistency with our system
                width = x2 - x1
                height = y2 - y1
                
                # Extract confidence score
                conf = float(box.conf[0])
                
                phone_data = {
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "confidence": conf
                }
                phones_detected.append(phone_data)
        
        # Cache the result for intermediate frames
        self._cached_phones = phones_detected
        self._cached_count = len(phones_detected)
                
        return phones_detected, len(phones_detected)

    def check_violations(self, phone_count: int) -> Tuple[bool, str]:
        """
        Analyzes the phone count to check for unauthorized devices.

        Args:
            phone_count (int): Number of detected cell phones.

        Returns:
            Tuple[bool, str]: A tuple containing violation status (True/False) and violation description.
        """
        if phone_count > 0:
            return True, f"Phone detected: {phone_count}"
        return False, "Normal"

    def draw_phones(self, frame: np.ndarray, phones_detected: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws bounding boxes and confidence scores around detected cell phones.

        Args:
            frame (np.ndarray): The BGR image frame from the webcam.
            phones_detected (List[Dict[str, Any]]): List of phones containing coordinate data.

        Returns:
            np.ndarray: The modified image frame with drawn annotations.
        """
        # [Performance P3] Draw directly on the frame passed in (no copy).
        # The caller is responsible for making a copy if the original is needed.
        # This saves ~1-2ms per frame by avoiding a full-resolution array copy.
        annotated_frame = frame

        for phone in phones_detected:
            x, y, w, h = phone["x"], phone["y"], phone["width"], phone["height"]
            conf = phone["confidence"]

            # Draw a red rectangle around the phone (red = critical violation)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the confidence score above the bounding box
            text = f"Phone {conf * 100:.1f}%"
            cv2.putText(annotated_frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return annotated_frame
