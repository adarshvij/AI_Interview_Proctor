"""
Warning and Risk-Score Management System

Tracks proctoring violations dynamically, records timestamps of events,
saves screenshot evidence of infractions, and aggregates candidate risk scores.
"""

import time
import os
import cv2
from typing import List, Dict, Any, Tuple
import numpy as np

class WarningSystem:
    """
    Acts as a central monitoring ledger during an exam. Processes raw detection
    signals, records occurrences, triggers screenshot saves, and calculates risk metrics.
    """

    def __init__(self, screenshot_dir: str = "screenshots", risk_growth_rate: int = 5):
        """
        Initializes the WarningSystem configuration.

        Args:
            screenshot_dir (str): Directory where violation screenshots are saved.
            risk_growth_rate (int): Incremental penalty value applied per logged infraction.
        """
        self.screenshot_dir = screenshot_dir
        self.risk_growth_rate = risk_growth_rate
        self.violation_log: List[Dict[str, Any]] = []
        self.risk_score: int = 0
        
        # Track consecutive frames of violations to filter transient noise
        self.consecutive_looking_away = 0
        self.consecutive_no_face = 0
        
        # Create output directories if they do not exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def update(self, face_count: int, is_looking_away: bool, phone_detected: bool, frame: np.ndarray) -> str:
        """
        Processes frame predictions, updates violation counts, and returns active status.

        Args:
            face_count (int): Number of faces visible in frame.
            is_looking_away (bool): True if candidate is looking away.
            phone_detected (bool): True if cell phone is in the scene.
            frame (np.ndarray): Original image frame to capture if violation occurred.

        Returns:
            str: Active proctoring classification status.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "Normal"

        # Check 1: Phone Usage (High Priority Violation)
        if phone_detected:
            self._log_violation("Phone Usage", "Mobile phone detected in frame", frame, timestamp)
            status = "Violation - Phone"

        # Check 2: Face Count Anomaly (Multiple or None)
        if face_count == 0:
            self.consecutive_no_face += 1
            if self.consecutive_no_face > 15:  # ~0.5 second threshold
                self._log_violation("No Face Detected", "Candidate is not in front of camera", frame, timestamp)
                status = "Violation - No Face"
        else:
            self.consecutive_no_face = 0

        if face_count > 1:
            self._log_violation("Multiple Faces", f"Detected {face_count} individuals", frame, timestamp)
            status = "Violation - Multiple People"

        # Check 3: Looking Away
        if is_looking_away:
            self.consecutive_looking_away += 1
            if self.consecutive_looking_away > 30:  # ~1 second threshold
                self._log_violation("Looking Away", "Candidate looking away from screen", frame, timestamp)
                status = "Violation - Distracted"
        else:
            self.consecutive_looking_away = 0

        self._calculate_risk_score()
        return status

    def _log_violation(self, violation_type: str, details: str, frame: np.ndarray, timestamp: str) -> None:
        """
        Records a violation incident, saves a visual screenshot, and appends to logs.
        """
        # Save screenshot
        filename = f"{violation_type.lower().replace(' ', '_')}_{int(time.time())}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        # Save placeholder screenshot on disk
        if frame is not None:
            cv2.imwrite(filepath, frame)
        
        # Append log entry
        self.violation_log.append({
            "timestamp": timestamp,
            "type": violation_type,
            "description": details,
            "screenshot": filepath
        })

    def _calculate_risk_score(self) -> None:
        """
        Calculates the overall risk percentage score (0-100) based on severity and count.
        """
        # Base multiplier logic: Phone counts for high risk, looking away is lower risk
        score = 0
        for log in self.violation_log:
            if log["type"] == "Phone Usage":
                score += 35
            elif log["type"] == "Multiple Faces":
                score += 25
            elif log["type"] == "No Face Detected":
                score += 15
            elif log["type"] == "Looking Away":
                score += 5
        
        self.risk_score = min(score, 100)

    def get_risk_score(self) -> int:
        """
        Retrieves the cumulative risk score of the session.

        Returns:
            int: Risk percentage (0-100).
        """
        return self.risk_score

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Returns all registered violation logs.

        Returns:
            List[Dict[str, Any]]: List of violations containing timestamps, types, and images.
        """
        return self.violation_log
