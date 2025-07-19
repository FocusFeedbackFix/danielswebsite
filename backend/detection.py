# backend/detection.py
import cv2
import threading
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# =========================
# Exception Classes
# =========================

class CapDetectionError(Exception):
    """Custom exception for cap detection errors."""
    pass

# =========================
# Data Classes
# =========================

class CapDetectionResult:
    """Stores detection results for a single cap."""
    def __init__(self, cap_number: int, box: Tuple[int, int, int, int], confidence: float = 1.0, color: str = ""):
        self.cap_number = cap_number
        self.box = box  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.color = color

    def as_dict(self):
        return {
            "cap": self.cap_number,
            "box": self.box,
            "confidence": self.confidence,
            "color": self.color
        }

# =========================
# CapDetector Class
# =========================

class CapDetector:
    def __init__(self, camera_index: int = 0, model_path: str = "path/to/your/yolo-weights.pt"):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.lock = threading.Lock()
        self.running = True
        self.last_frame = None
        self.last_detected_caps: List[CapDetectionResult] = []
        self.last_annotated = None
        self.frame_thread = threading.Thread(target=self._frame_updater, daemon=True)
        self.frame_thread.start()
        self.model = YOLO(model_path)
        self.class_map = self._load_class_map()
        logging.info("CapDetector initialized with YOLO model.")

    def __del__(self):
        self.stop()
        logging.info("CapDetector destroyed.")

    def stop(self): 
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        logging.info("Video capture released.")

    def _frame_updater(self):
        """Continuously updates the latest frame in a background thread."""
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.last_frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS

    def get_frame(self) -> Optional[Any]:
        """Returns the latest frame."""
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
            else:
                return None

    def _load_class_map(self) -> Dict[int, str]:
        """
        Loads a mapping from class index to cap number or color.
        Modify this method to match your YOLO model's class mapping.
        """
        # Example: {0: "1", 1: "2", ..., 12: "13"}
        return {i: str(i+1) for i in range(13)}

    def detect_caps(self, frame) -> List[CapDetectionResult]:
        """
        Detects cap numbers in the given frame using YOLO.
        Returns a list of CapDetectionResult.
        """
        try:
            results = self.model(frame, verbose=False)
            detected_caps = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_idx = int(box.cls[0])
                cap_num = int(self.class_map.get(class_idx, -1))
                conf = float(box.conf[0])
                color = self._infer_color(frame, (x1, y1, x2, y2))
                if cap_num != -1:
                    detected_caps.append(CapDetectionResult(cap_num, (x1, y1, x2, y2), conf, color))
            logging.debug(f"Detected caps: {[d.as_dict() for d in detected_caps]}")
            return detected_caps
        except Exception as e:
            logging.error(f"YOLO detection failed: {e}")
            raise CapDetectionError(f"YOLO detection failed: {e}")

    def _infer_color(self, frame, box: Tuple[int, int, int, int]) -> str:
        """
        Dummy color inference: returns 'blue' or 'white' based on mean color.
        Replace with your own logic if needed.
        """
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"
        mean_color = roi.mean(axis=(0, 1))
        if mean_color[0] > mean_color[2]:
            return "blue"
        else:
            return "white"

    def annotate_frame(self, frame, detected_caps: List[CapDetectionResult]) -> Any:
        """
        Draws bounding boxes and labels for detected caps on the frame.
        """
        for det in detected_caps:
            x1, y1, x2, y2 = det.box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Cap {det.cap_number} {det.color} ({det.confidence*100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def get_detection_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary with the latest detection info.
        """
        frame = self.get_frame()
        if frame is None:
            return {"caps": [], "timestamp": time.time()}
        detected_caps = self.detect_caps(frame)
        self.last_detected_caps = detected_caps
        return {
            "caps": [det.cap_number for det in detected_caps],
            "boxes": [det.box for det in detected_caps],
            "confidences": [det.confidence for det in detected_caps],
            "colors": [det.color for det in detected_caps],
            "timestamp": time.time()
        }

    def get_annotated_frame(self) -> Optional[Any]:
        """
        Returns the latest frame with annotations.
        """
        frame = self.get_frame()
        if frame is None:
            return None
        detected_caps = self.detect_caps(frame)
        annotated = self.annotate_frame(frame, detected_caps)
        self.last_annotated = annotated
        return annotated

    def save_frame(self, path: str) -> bool:
        """
        Saves the latest frame to the given path.
        """
        frame = self.get_frame()
        if frame is not None:
            cv2.imwrite(path, frame)
            logging.info(f"Frame saved to {path}")
            return True
        logging.warning("No frame to save.")
        return False

    def save_annotated(self, path: str) -> bool:
        """
        Saves the latest annotated frame to the given path.
        """
        annotated = self.get_annotated_frame()
        if annotated is not None:
            cv2.imwrite(path, annotated)
            logging.info(f"Annotated frame saved to {path}")
            return True
        logging.warning("No annotated frame to save.")
        return False

    def get_video_properties(self) -> Dict[str, Any]:
        """
        Returns properties of the video capture device.
        """
        props = {
            "width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": self.cap.get(cv2.CAP_PROP_FOURCC),
            "frame_count": self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
        }
        logging.info(f"Video properties: {props}")
        return props

    def set_camera_index(self, index: int):
        """
        Changes the camera index and reinitializes the capture.
        """
        self.stop()
        self.camera_index = index
        self.cap = cv2.VideoCapture(index)
        self.running = True
        self.frame_thread = threading.Thread(target=self._frame_updater, daemon=True)
        self.frame_thread.start()
        logging.info(f"Camera index changed to {index}")

    def is_camera_opened(self) -> bool:
        return self.cap.isOpened()

    def restart_camera(self):
        self.stop()
        time.sleep(0.5)
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        self.frame_thread = threading.Thread(target=self._frame_updater, daemon=True)
        self.frame_thread.start()
        logging.info("Camera restarted.")

    def get_dummy_detection_stats(self) -> Dict[str, Any]:
        """
        Returns dummy stats for testing.
        """
        return {
            "caps_detected": len(self.last_detected_caps),
            "last_caps": [det.cap_number for det in self.last_detected_caps],
            "timestamp": time.time()
        }

    def get_last_annotated(self) -> Optional[Any]:
        """Returns the last annotated frame."""
        return self.last_annotated

    def get_last_detected_caps(self) -> List[CapDetectionResult]:
        """Returns the last detected caps."""
        return self.last_detected_caps

# =========================
# Singleton detector instance
# =========================

detector = CapDetector(model_path="path/to/your/yolo-weights.pt")

# =========================
# API/Utility Functions
# =========================

def gen_frames():
    """
    Generator function for streaming annotated video frames (MJPEG).
    """
    while True:
        frame = detector.get_annotated_frame()
        if frame is None:
            logging.warning("No frame available for streaming.")
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Failed to encode frame.")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30 FPS

def detect_caps():
    """
    Returns the latest detected cap numbers as a list.
    """
    info = detector.get_detection_info()
    return info["caps"]

def get_detection_stats():
    """
    Returns detection statistics.
    """
    return detector.get_dummy_detection_stats()

def save_latest_frame(path: str) -> bool:
    """
    Saves the latest frame to disk.
    """
    return detector.save_frame(path)

def save_latest_annotated(path: str) -> bool:
    """
    Saves the latest annotated frame to disk.
    """
    return detector.save_annotated(path)

def get_video_info() -> Dict[str, Any]:
    """
    Returns video capture properties.
    """
    return detector.get_video_properties()

def change_camera(index: int):
    """
    Changes the camera index.
    """
    detector.set_camera_index(index)

def restart_camera():
    """
    Restarts the camera.
    """
    detector.restart_camera()

def get_last_annotated_frame_bytes() -> Optional[bytes]:
    """
    Returns the last annotated frame as JPEG bytes.
    """
    frame = detector.get_last_annotated()
    if frame is None:
        return None
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    return buffer.tobytes()

def get_last_detection_json() -> Dict[str, Any]:
    """
    Returns the last detection as a JSON-serializable dict.
    """
    return {
        "caps": [det.cap_number for det in detector.get_last_detected_caps()],
        "boxes": [det.box for det in detector.get_last_detected_caps()],
        "confidences": [det.confidence for det in detector.get_last_detected_caps()],
        "colors": [det.color for det in detector.get_last_detected_caps()],
        "timestamp": time.time()
    }

def get_camera_status() -> Dict[str, Any]:
    """
    Returns camera status and health info.
    """
    return {
        "opened": detector.is_camera_opened(),
        "properties": detector.get_video_properties()
    }

def set_and_restart_camera(index: int):
    """
    Changes camera index and restarts camera.
    """
    change_camera(index)
    restart_camera()
    logging.info(f"Camera changed and restarted to index {index}")

def get_cap_detection_error_example() -> Dict[str, Any]:
    """
    Returns an example error response.
    """
    try:
        raise CapDetectionError("Example error for testing.")
    except CapDetectionError as e:
        return {"error": str(e), "type": "CapDetectionError"}

# =========================
# Advanced/Expansion: Batch Processing, Video File, etc.
# =========================

def detect_caps_in_video(video_path: str) -> List[Dict[str, Any]]:
    """
    Runs detection on a video file and returns a list of detection results per frame.
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            detected = detector.detect_caps(frame)
            results.append({
                "frame": frame_idx,
                "caps": [d.as_dict() for d in detected]
            })
        except Exception as e:
            logging.error(f"Detection failed on frame {frame_idx}: {e}")
        frame_idx += 1
    cap.release()
    return results

def save_detection_results_to_file(results: List[Dict[str, Any]], path: str):
    """
    Saves detection results to a JSON file.
    """
    import json
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Detection results saved to {path}")

# =========================
# Main for Testing/Debugging
# =========================

if __name__ == "__main__":
    print("Testing CapDetector backend module...")
    print("Video properties:", get_video_info())
    print("Detection stats:", get_detection_stats())
    save_latest_frame("test_frame.jpg")
    save_latest_annotated("test_annotated.jpg")
    # Batch process a video (optional)
    # results = detect_caps_in_video("test_video.mp4")
    # save_detection_results_to_file(results, "results.json")
    print("Done.")

# End of detection.py