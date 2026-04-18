import cv2
from tracker_utils import DotTracker, track_dot, display_dot_info
from utils import save_json, VideoManager, load_json
import matplotlib.pyplot as plt
import os  # Import os module

# Define paths for saving data
CONFIG_PATH = "data/config.json"
COORDINATES_PATH = "data/coordinates.json"
FFT_PATH = "data/fft.json"

def train_fft_model(video_manager, window_size=300, sampling_rate=30, debug=False, duration=10):
    """Train FFT model by processing a video to extract motion data."""
    # Get video properties
    fps = video_manager.cap.get(cv2.CAP_PROP_FPS)
    resolution = (int(video_manager.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_manager.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Load colors dynamically from configuration file
    config = load_json(CONFIG_PATH)
    colors = config.get("colors", ["red"])  # Default to red if not specified

    # Initialize trackers and buffers
    trackers = {color: DotTracker(json_path=COORDINATES_PATH, buffer_size=window_size) for color in colors}

    # Buffers for raw data
    raw_data = {color: {"x_series": [], "y_series": []} for color in colors}

    # Clear the JSON file at the start of each training session
    save_json({color: raw_data[color] for color in colors}, COORDINATES_PATH)

    total_frames = int(duration * sampling_rate)  # Calculate the total number of frames to sample
    frame_count = 0  # Initialize a frame counter

    # Ensure config.json exists and overwrite its content
    config_data = {
        "fps": sampling_rate,
        "window_size": window_size,
        "colors": colors,
        "resolution": [
            int(video_manager.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_manager.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ]
    }
    save_json(config_data, CONFIG_PATH)

    while frame_count < total_frames:
        ret, frame = video_manager.read_frame()
        if not ret:
            break

        frame_count += 1  # Increment the frame counter

        for color in colors:
            position = track_dot(frame, color)  # Use track_dot to find the position
            trackers[color].update(position)  # Update the tracker with the position

            # Append the position to raw_data for every frame
            if position:
                raw_data[color]["x_series"].append(position[0])
                raw_data[color]["y_series"].append(position[1])
            else:
                raw_data[color]["x_series"].append(None)
                raw_data[color]["y_series"].append(None)

            # Ensure dot info is displayed
            frame = display_dot_info(frame, position, color)

        # Show the frame with the dot overlay
        cv2.imshow("Training Video", frame)

        # Break the loop if 'q' is pressed
        if video_manager.check_keyboard_break(key='q'):
            print("Training interrupted by user.")
            break

    # Close the OpenCV window after the loop ends
    video_manager.close_window()

    # Save the accumulated raw_data to the JSON file
    save_json(raw_data, COORDINATES_PATH)

    # Plot raw x and y data
    if debug:
        for color, data in raw_data.items():
            plt.figure(figsize=(10, 6))
            plt.plot(data['x_series'], label=f'{color} X')
            plt.plot(data['y_series'], label=f'{color} Y')
            plt.title(f'Raw Data for {color}')
            plt.xlabel('Frame')
            plt.ylabel('Position')
            plt.legend()
            plt.grid(True)
            plt.show()

    print("Training data saved successfully.")

def log_training_data(raw_data, filename):
    """Log training data to JSON file."""
    save_json(raw_data, filename)

def main():
    """Entry point for training motion tracking."""
    ## #video_manager = VideoManager("videos/train.mp4")
    video_manager = VideoManager(0)
    try:
        train_fft_model(video_manager, window_size=300, sampling_rate=30, debug=True, duration=10)
    finally:
        video_manager.release()

if __name__ == "__main__":
    main()
