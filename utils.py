import os
import json
import numpy as np
import cv2
import threading
from scipy.signal import correlate
from sklearn.metrics.pairwise import cosine_similarity

# JSON Utilities
def save_json(data, filename):
    """Save data to a JSON file, creating the file if it doesn't exist."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """Utility function to load data from a JSON file."""
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        print(f"Warning: {filename} is empty or does not exist. Returning an empty dictionary.")
        return {}
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}. Returning an empty dictionary.")
        return {}

# Video Utilities
class VideoManager:
    """Manages a shared video capture instance"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video source {source}.")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.isOpened() else 0
        if self.fps <= 0:
            raise ValueError("Unable to determine FPS from video source.")
        # Initialize a threading lock to ensure thread-safe access to the video capture instance
        self.lock = threading.Lock()

    def read_frame(self):
        """Read a frame from the video or camera."""
        # Acquire the lock before accessing the shared video capture instance
        with self.lock:
            ret, frame = self.cap.read()
        # The lock is automatically released after the block
        return ret, frame

    def release(self):
        """Release the video capture resource."""
        # Acquire the lock to ensure no other thread is using the video capture instance
        with self.lock:
            self.cap.release()
        # The lock ensures that the release operation is thread-safe

    def check_keyboard_break(self, key='q'):
        """Check if a specific key is pressed to break the loop."""
        if cv2.waitKey(1) & 0xFF == ord(key):
            return True
        return False

    def close_window(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()

# Data Processing Utilities
def interpolate_nan_values(series):
    """Interpolate NaN values in a series."""
    series = np.array(series, dtype=np.float32)
    nans = np.isnan(series)
    indices = np.arange(len(series))
    if np.any(nans):
        series[nans] = np.interp(indices[nans], indices[~nans], series[~nans])
    return np.round(series, 1)  # Round to 1 decimal place

def cross_correlation(live_series, training_series, window_size):
    """Find the best matching segment of N frames within the buffer using cross-correlation."""
    correlation = correlate(live_series, training_series, mode='valid')
    best_match_index = np.argmax(correlation)
    return best_match_index

def calculate_similarity_metrics(segment, training_segment):
    """Calculate similarity metrics (cosine and Euclidean distance) between two segments."""
    segment = np.array(segment, dtype=np.float32)
    training_segment = np.array(training_segment, dtype=np.float32)

    # Normalize the segments
    segment_norm = np.linalg.norm(segment)
    training_norm = np.linalg.norm(training_segment)

    if segment_norm > 0 and training_norm > 0:
        cosine_sim = cosine_similarity([segment], [training_segment])[0][0]
    else:
        cosine_sim = 0.0

    euclidean_dist = np.linalg.norm(segment - training_segment)

    return {
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist
    }

# Overlay Utilities
def overlay_dot_info(frame, position, color):
    """Overlay the dot information on the video frame."""
    # Convert color name to BGR format
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }
    bgr_color = color_map.get(color, (255, 255, 255))  # Default to white if color is not recognized

    if position:
        cv2.circle(frame, position, 10, bgr_color, 2)
        text_position = (position[0] + 15, position[1] - 15)
        cv2.putText(frame, f"Coordinates: {position}", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {color}", (text_position[0], text_position[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
