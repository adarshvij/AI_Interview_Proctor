"""
Warning System Module

Acts as the central decision-making hub. It aggregates inputs from the Face,
Gaze, and Phone detectors, maintains violation counters, and calculates an
overall risk level (Low, Medium, High).

[Bug fix B3] Uses *event-based* (edge-triggered) counting instead of per-frame
counting. Counters increment only when a violation state *begins* (transitions
from clean → violation), not on every frame where the violation persists.
This prevents risk scores from skyrocketing within seconds of a sustained
violation (e.g., holding a phone for 2s no longer adds ~30 counts × 50 weight).
"""

from typing import Dict, Any

class WarningSystem:
    """
    Maintains state and calculates risk levels based on cumulative violations.
    """

    def __init__(self):
        """
        Initializes the violation counters, thresholds, and previous-state trackers.
        """
        # We track the count of distinct violation *events* (state transitions)
        self.counters = {
            "multiple_faces": 0,
            "phone_usage": 0,
            "looking_away": 0,
            "no_face": 0
        }
        
        # [Bug fix B3] Previous-frame state for edge detection.
        # We only increment a counter when the state changes from False → True.
        self._prev_state = {
            "no_face": False,
            "multiple_faces": False,
            "phone_usage": False,
            "looking_away": False
        }
        
        self.total_risk_score = 0

    def update(self, face_count: int, looking_away: bool, phone_detected: bool) -> Dict[str, Any]:
        """
        Updates the counters based on the latest frame's data and calculates the risk level.

        [Bug fix B3] Only increments a counter on the *rising edge* — i.e., when
        the violation was absent last frame but is present this frame. Sustained
        violations (e.g., phone held for 5 seconds) count as a single event.

        Args:
            face_count (int): Number of detected faces from FaceDetector.
            looking_away (bool): True if candidate is looking away from GazeDetector.
            phone_detected (bool): True if a phone is detected from PhoneDetector.

        Returns:
            Dict[str, Any]: A dictionary containing current counters and the calculated risk level.
        """
        # Derive current boolean states from raw inputs
        current_no_face = (face_count == 0)
        current_multiple = (face_count > 1)

        # [Bug fix B3] Edge-triggered increment: only count when state transitions
        # from False → True (rising edge). This ensures "phone held for 10 seconds"
        # = 1 event, not 150+ frame-level increments.
        if current_no_face and not self._prev_state["no_face"]:
            self.counters["no_face"] += 1
        if current_multiple and not self._prev_state["multiple_faces"]:
            self.counters["multiple_faces"] += 1
        if phone_detected and not self._prev_state["phone_usage"]:
            self.counters["phone_usage"] += 1
        if looking_away and not self._prev_state["looking_away"]:
            self.counters["looking_away"] += 1

        # Update previous state for next frame's edge detection
        self._prev_state["no_face"] = current_no_face
        self._prev_state["multiple_faces"] = current_multiple
        self._prev_state["phone_usage"] = phone_detected
        self._prev_state["looking_away"] = looking_away

        # Calculate Risk Level
        risk_level = self._calculate_risk_level()

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
        
        [Bug fix B3] Thresholds adjusted for event-based counting.
        Previously calibrated for frame-level counts (hundreds per minute),
        now calibrated for distinct events (single-digit counts per session).
        
        Returns:
            str: "Low", "Medium", or "High"
        """
        # Assign different "weights" to different infractions based on severity.
        # These weights reflect event-level severity (each count = one distinct incident).
        score = 0
        score += self.counters["phone_usage"] * 50     # Critical violation
        score += self.counters["multiple_faces"] * 30  # High violation
        score += self.counters["no_face"] * 10         # Medium violation
        score += self.counters["looking_away"] * 5     # Minor violation (unless repeated heavily)

        self.total_risk_score = score

        # [Bug fix B3] Thresholds lowered for event-based counting:
        # - Low:    < 3 events worth of score (e.g., 1 phone event = 50, borderline)
        # - Medium: < 8 events worth of score
        # - High:   >= 8 events worth
        if score < 50:
            return "Low"
        elif score < 150:
            return "Medium"
        else:
            return "High"

