"""
Warning System Module

Acts as the central decision-making hub. It aggregates inputs from the Face,
Gaze, and Phone detectors, maintains violation counters, and calculates an
overall risk level (Low, Medium, High).
"""

from typing import Dict, Any

class WarningSystem:
    """
    Maintains state and calculates risk levels based on cumulative violations.
    """

    def __init__(self):
        """
        Initializes the violation counters and thresholds.
        """
        # We track the exact count of frames/occurrences for each rule violation
        self.counters = {
            "multiple_faces": 0,
            "phone_usage": 0,
            "looking_away": 0,
            "no_face": 0
        }
        
        self.total_risk_score = 0

    def update(self, face_count: int, looking_away: bool, phone_detected: bool) -> Dict[str, Any]:
        """
        Updates the counters based on the latest frame's data and calculates the risk level.

        Args:
            face_count (int): Number of detected faces from FaceDetector.
            looking_away (bool): True if candidate is looking away from GazeDetector.
            phone_detected (bool): True if a phone is detected from PhoneDetector.

        Returns:
            Dict[str, Any]: A dictionary containing current counters and the calculated risk level.
        """
        # 1. Update Counters based on raw detection inputs
        if face_count == 0:
            self.counters["no_face"] += 1
        elif face_count > 1:
            self.counters["multiple_faces"] += 1
            
        if phone_detected:
            self.counters["phone_usage"] += 1
            
        if looking_away:
            self.counters["looking_away"] += 1

        # 2. Calculate Risk Level
        risk_level = self._calculate_risk_level()

        # 3. Return the current snapshot state for the UI
        # [Bug fix B2] Return a COPY of counters, not a reference.
        # Without .copy(), the dict in session_state is the same object as
        # self.counters, so Streamlit cannot detect changes between reruns.
        return {
            "counters": self.counters.copy(),
            "risk_level": risk_level
        }

    def _calculate_risk_level(self) -> str:
        """
        Calculates the risk level based on the weighted severity of violations.
        
        Returns:
            str: "Low", "Medium", or "High"
        """
        # Assign different "weights" to different infractions based on severity
        score = 0
        score += self.counters["phone_usage"] * 50     # Critical violation
        score += self.counters["multiple_faces"] * 30  # High violation
        score += self.counters["no_face"] * 5          # Medium violation
        score += self.counters["looking_away"] * 2     # Minor violation (unless repeated heavily)

        self.total_risk_score = score

        # Determine the categorical risk level
        if score < 50:
            return "Low"
        elif score < 150:
            return "Medium"
        else:
            return "High"
