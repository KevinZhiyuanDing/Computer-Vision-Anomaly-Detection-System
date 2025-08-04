# Tracker Utilities
import cv2
import numpy as np
from collections import deque
from utils import save_json, interpolate_nan_values
import matplotlib.pyplot as plt
from utils import VideoManager

# Define HSV ranges for colored dots - simplified to focus on red only
COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ],
}

def track_dot(frame, color):
    """Track a single colored dot in the frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for the specified color
    ranges = COLOR_RANGES[color]
    mask = None
    for lower, upper in ranges:
        current_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

    # Debugging: Show the mask
    ## cv2.imshow(f"{color} Mask", mask)

    # Find contours and get the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        if radius > 5:  # Minimum size threshold
            return (int(x), int(y))

    return None
    
class DotTracker:
    """Tracks dot positions over time with a fixed buffer size."""
    def __init__(self, json_path, buffer_size=128, debug=False):
        self.positions = deque(maxlen=buffer_size)
        self.json_path = json_path
        self.debug = debug

    def update(self, position):
        """Add a new position to the tracker."""
        self.positions.append(position)
        if self.debug:
            print(f"Updated tracker with position: {position}")

    def get_positions(self):
        """Get all currently tracked positions."""
        return list(self.positions)

    def get_x_series(self):
        """Extract x-coordinates from positions."""
        x_series = []
        for pos in self.positions:
            if pos is not None:
                x_series.append(pos[0])
            else:
                x_series.append(np.nan)
        return x_series

    def get_y_series(self):
        """Extract y-coordinates from positions."""
        y_series = []
        for pos in self.positions:
            if pos is not None:
                y_series.append(pos[1])
            else:
                y_series.append(np.nan)
        return y_series

    def log_to_json(self):
        """Log the x and y series to a JSON file."""
        data = {
            "x_series": interpolate_nan_values(self.get_x_series()),
            "y_series": interpolate_nan_values(self.get_y_series())
        }
        save_json(data, self.json_path)

def log_to_json(tracker, filename):
    """Log tracker data to JSON file."""
    data = {
        "x_series": interpolate_nan_values(tracker.get_x_series()),
        "y_series": interpolate_nan_values(tracker.get_y_series())
    }
    save_json(data, filename)

def display_dot_info(frame, position, color):
    """Display dot information (coordinates and color) on the frame."""
    if position:
        cv2.circle(frame, position, 10, (0, 0, 255), 2)
        text_position = (position[0] + 15, position[1] - 15)
        cv2.putText(frame, f"Coordinates: {position}", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {color}", (text_position[0], text_position[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def display_video_with_overlay(video_manager, trackers, colors, debug=False):
    """Display video with colored dot overlay and annotations."""
    while True:
        ret, frame = video_manager.read_frame()
        if not ret:
            break

        for color in colors:
            # Dynamically update the position for the current frame
            position = track_dot(frame, color)
            if position:
                trackers[color].update(position)
                frame = display_dot_info(frame, position, color)

        cv2.imshow("Debug Video", frame)

def plot_results(raw_data, fft_data):
    """Plot raw X, Y data and FFT results."""
    for color, data in raw_data.items():
        plt.figure(figsize=(12, 6))

        # Plot raw X and Y data
        plt.subplot(2, 1, 1)
        plt.plot(data['x_series'], label=f'{color} X')
        plt.plot(data['y_series'], label=f'{color} Y')
        plt.title(f'Raw Data for {color}')
        plt.legend()

        # Plot FFT results
        plt.subplot(2, 1, 2)
        x_freq = fft_data[color]['x']['frequencies']
        x_amp = fft_data[color]['x']['amplitudes']
        y_freq = fft_data[color]['y']['frequencies']
        y_amp = fft_data[color]['y']['amplitudes']

        # Ensure frequencies and amplitudes have the same length
        x_len = min(len(x_freq), len(x_amp))
        y_len = min(len(y_freq), len(y_amp))

        plt.plot(x_freq[:x_len], x_amp[:x_len], label=f'{color} X FFT')
        plt.plot(y_freq[:y_len], y_amp[:y_len], label=f'{color} Y FFT')
        plt.title(f'FFT Results for {color}')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    """Entry point for testing tracker utilities."""
    video_manager = VideoManager("videos/test.mp4")
    colors = ["red"]
    trackers = {color: DotTracker(json_path="data/test_coordinates.json") for color in colors}

    while True:
        ret, frame = video_manager.read_frame()
        if not ret:
            break

        for color in colors:
            position = track_dot(frame, color)
            trackers[color].update(position)
            frame = display_dot_info(frame, position, color)

        cv2.imshow("Test Video", frame)
        if video_manager.check_keyboard_break(key='q'):
            break

    video_manager.close_window()

if __name__ == "__main__":
    main()
