"""
Webcam Stream Controller

Provides thread-safe initialization, capture, and closing of webcam resources.
Uses OpenCV for camera communication and includes provisions for streaming.

[Performance P1] Uses a background daemon thread for continuous frame capture.
The main processing loop calls read() which returns the *latest* frame instantly
(non-blocking), eliminating the 30-50ms synchronous cv2.VideoCapture.read() call
that previously dominated each loop iteration.
"""

import cv2
import threading
from typing import Optional, Tuple

class WebcamStream:
    """
    Handles camera interaction, frame capturing, and configurations in a structured,
    production-ready manner. Uses a background thread for non-blocking capture.
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
        # [Performance P1] Handle for the background capture thread.
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """
        Initializes the camera and launches the background capture thread.

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

        # [Performance P1] Launch a daemon thread that continuously grabs frames.
        # This decouples camera I/O (which blocks on driver timing, typically ~33ms
        # at 30 FPS) from the processing loop, so the main thread never waits.
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def _capture_loop(self) -> None:
        """
        [Performance P1] Background thread: continuously reads frames from the
        camera and stores the latest one. Runs until is_running is set to False.
        """
        while self.is_running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.latest_frame = frame
            # If read() fails (e.g. camera disconnected), we keep trying
            # rather than crashing — the main loop will see None and handle it.

    def read(self) -> Optional[cv2.Mat]:
        """
        Returns the latest frame captured by the background thread.
        
        [Performance P1] This is non-blocking — it returns immediately with
        whatever frame the background thread last captured, instead of waiting
        30-50ms for the next camera read.

        Returns:
            Optional[cv2.Mat]: The BGR image array if available, or None.
        """
        if not self.is_running:
            return None
            
        with self._lock:
            return self.latest_frame

    def release(self) -> None:
        """
        Signals the capture thread to stop, waits for it, and releases resources.
        """
        # Signal the thread to stop first
        self.is_running = False

        # [Performance P1] Wait for the capture thread to finish before releasing
        # the VideoCapture — otherwise the thread could call cap.read() on a
        # released object.
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
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