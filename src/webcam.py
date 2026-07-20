"""
Webcam Stream Controller

Provides thread-safe initialization, capture, and closing of webcam resources.
Uses OpenCV for camera communication and includes provisions for streaming.
"""

import cv2
import threading
from typing import Optional, Tuple

class WebcamStream:
    """
    Handles camera interaction, frame capturing, and configurations in a structured,
    production-ready manner. Provides placeholders for thread-safe multi-threaded capturing.
    """
    
    def __init__(self, src: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """
        Initializes the webcam stream configuration.

        Args:
            src (int): Camera hardware index (default: 0).
            resolution (Tuple[int, int]): Target capture width and height.
        """
        self.src = src
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._lock = threading.Lock()
        self.latest_frame = None

    def start(self) -> bool:
        """
        Initializes and opens the camera stream.

        Returns:
            bool: True if the camera successfully opened, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            return False
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.is_running = True
        return True

    def read(self) -> Optional[cv2.Mat]:
        """
        Grabs the latest frame from the video capture.

        Returns:
            Optional[cv2.Mat]: The BGR image array if successful, or None.
        """
        if not self.is_running or self.cap is None:
            return None
            
        with self._lock:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                return frame
            return None

    def release(self) -> None:
        """
        Releases the webcam video capture resource and resets the state.
        """
        with self._lock:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.latest_frame = None
            

if __name__ == "__main__":
    webcam = WebcamStream()

    if not webcam.start():
        print("Error: Could not open webcam.")
        raise SystemExit

    print("Webcam started. Press 'q' to quit.")

    while True:
        frame = webcam.read()

        if frame is None:
            print("Failed to capture frame.")
            break

        cv2.imshow("AI Interview Proctor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()