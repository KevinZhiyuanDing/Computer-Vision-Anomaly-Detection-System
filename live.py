import cv2
import numpy as np
import json
from collections import deque
from tracker_utils import DotTracker, interpolate_nan_values, track_dot
from fft_utils import process_fft, normalize_fft_magnitude
from utils import VideoManager, cross_correlation, calculate_similarity_metrics, overlay_dot_info, load_json
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import os  # Importing os for file existence checks
from scipy.signal import correlate
from sklearn.decomposition import PCA
import pickle  # Importing pickle for loading PCA model

# Define paths for loading data
CONFIG_PATH = "data/config.json"
COORDINATES_PATH = "data/coordinates.json"
FFT_PATH = "data/fft.json"
CYCLE_PATH = "data/cycle.json"

def validate_configuration(config, video_manager):
    """Validate configuration against live video properties."""
    live_fps = video_manager.cap.get(cv2.CAP_PROP_FPS)
    live_resolution = (
        int(video_manager.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_manager.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    if live_fps != config.get("fps", 30):
        raise ValueError(f"Mismatch in FPS: Live video FPS ({live_fps}) does not match training FPS ({config.get('fps', 30)}).")

    if live_resolution != tuple(config.get("resolution", (640, 480))):
        raise ValueError(
            f"Mismatch in resolution: Live video resolution ({live_resolution}) does not match training resolution ({config.get('resolution')})."
        )

def process_live_series(sliding_window, avg_cycle, color, cycle_window_length, x_pca, y_pca, threshold):
    """Process live series for cross-correlation, FFT, PCA, and anomaly detection."""
    live_series_x = [position[0] for position in sliding_window[color]]  # Extract x values
    live_series_x = interpolate_nan_values(live_series_x)

    live_series_y = [position[1] for position in sliding_window[color]]  # Extract y values
    live_series_y = interpolate_nan_values(live_series_y)

    # Compute cross-correlation values for x
    cross_corr_values_x = np.correlate(live_series_x, avg_cycle[color]["x"], mode='full')
    x_best_lag = np.argmax(cross_corr_values_x) - (len(avg_cycle[color]["x"]) - 1)
    print(f"Best Lag for {color} (X): {x_best_lag}")

    # Extract the cycle window length live window based on the offset for x
    x_start_index = max(0, x_best_lag)
    x_end_index = x_start_index + cycle_window_length
    if x_end_index > len(live_series_x):
        raise ValueError(f"Cross-correlated window exceeds live series length for {color} (X).")
    x_live_window = live_series_x[x_start_index:x_end_index]

    # Compute cross-correlation values for y
    cross_corr_values_y = np.correlate(live_series_y, avg_cycle[color]["y"], mode='full')
    y_best_lag = np.argmax(cross_corr_values_y) - (len(avg_cycle[color]["y"]) - 1)
    print(f"Best Lag for {color} (Y): {y_best_lag}")

    # Extract the cycle window length live window based on the offset for y
    y_start_index = max(0, y_best_lag)
    y_end_index = y_start_index + cycle_window_length
    if y_end_index > len(live_series_y):
        raise ValueError(f"Cross-correlated window exceeds live series length for {color} (Y).")
    y_live_window = live_series_y[y_start_index:y_end_index]

    # Apply FFT
    fft_result_x = process_fft(x_live_window, cycle_window_length, len(avg_cycle[color]["x"]))
    fft_result_y = process_fft(y_live_window, cycle_window_length, len(avg_cycle[color]["y"]))

    # Normalize FFT
    fft_norm_x = normalize_fft_magnitude(fft_result_x)
    fft_norm_y = normalize_fft_magnitude(fft_result_y)

    # Use trained PCA models
    if not hasattr(x_pca, 'components_') or not hasattr(y_pca, 'components_'):
        raise ValueError("PCA models are not trained. Ensure PCA components are loaded correctly.")

    proj_x = x_pca.transform([fft_norm_x])
    recon_x = x_pca.inverse_transform(proj_x)
    proj_y = y_pca.transform([fft_norm_y])
    recon_y = y_pca.inverse_transform(proj_y)
    print(f"PCA projection and reconstruction completed for {color} (X and Y).")

    # Compute reconstruction error
    error_x = np.linalg.norm(fft_norm_x - recon_x)
    error_y = np.linalg.norm(fft_norm_y - recon_y)
    print(f"Reconstruction error for {color} (X): {error_x}")
    print(f"Reconstruction error for {color} (Y): {error_y}")

    # Helper function to compute normalized cross-correlation
    def normalized_cross_correlation(a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0:
            a /= norm_a
        if norm_b > 0:
            b /= norm_b
        corr = correlate(a, b, mode='valid')
        return corr

    # Compute normalized cross-correlation values for X and Y
    normalized_cross_corr_x = normalized_cross_correlation(live_series_x, avg_cycle[color]["x"])
    normalized_cross_corr_y = normalized_cross_correlation(live_series_y, avg_cycle[color]["y"])

    # Determine anomaly based on normalized cross-correlation threshold
    max_corr_x = np.max(normalized_cross_corr_x)
    max_corr_y = np.max(normalized_cross_corr_y)

    if max_corr_x < 0.5 or max_corr_y < 0.5:
        print(f"Anomaly detected for {color}! Normalized cross-correlation values: X={max_corr_x}, Y={max_corr_y}")
    else:
        print(f"Motion is normal for {color}. Normalized cross-correlation values: X={max_corr_x}, Y={max_corr_y}")

    def plot_overlay(avg_cycle, x_live_window, y_live_window, color):
        """Plot overlay of live window and average cycle."""
        plt.figure(figsize=(10, 5))
        plt.plot(avg_cycle[color]["x"], label="Average Cycle X", linestyle="--")
        plt.plot(avg_cycle[color]["y"], label="Average Cycle Y", linestyle="--")
        plt.plot(x_live_window, label="Live Window X", alpha=0.7)
        plt.plot(y_live_window, label="Live Window Y", alpha=0.7)
        plt.legend()
        plt.title(f"Overlay of Live Window and Average Cycle for {color}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    plot_overlay(avg_cycle, x_live_window, y_live_window, color)

def run_live_detection(video_manager, config, training_coordinates, pca_models, cycle_window, avg_cycle):
    """Run live detection for motion tracking and anomaly detection."""
    sampling_rate = config.get("fps", 30)
    colors = config.get("colors", ["red"])

    # Initialize trackers and buffers
    trackers = {color: DotTracker(json_path=COORDINATES_PATH) for color in colors}
    sliding_window = {color: deque(maxlen=int(2 * cycle_window[color]["x"])) for color in colors}
    increment_counter = {color: 0 for color in colors}

    try:
        while True:
            ret, frame = video_manager.read_frame()
            if not ret:
                break

            for color in colors:
                position = track_dot(frame, color)
                trackers[color].update(position)

                if position:
                    # Ensure sliding_window contains tuples (x, y)
                    if isinstance(position, (list, tuple)) and len(position) == 2:
                        sliding_window[color].append(position)
                    else:
                        sliding_window[color].append((position, np.nan))  # Default y to NaN if missing
                else:
                    sliding_window[color].append((np.nan, np.nan))  # Default both x and y to NaN

                increment_counter[color] += 1
                if increment_counter[color] % 10 == 0:
                    print(f"Sliding window size for {color}: {len(sliding_window[color])}")

                frame = overlay_dot_info(frame, position, color)

            cv2.imshow("Live Tracking", frame)

            if video_manager.check_keyboard_break(key='q'):
                print("Live detection interrupted by user.")
                break

            for color in colors:
                if len(sliding_window[color]) == sliding_window[color].maxlen:
                    if color in pca_models:
                        x_pca, y_pca = pca_models[color]
                        threshold = 0.1  # Example threshold, replace with actual value
                        process_live_series(sliding_window, avg_cycle, color, cycle_window[color]["x"], x_pca, y_pca, threshold)
                    else:
                        print(f"PCA models not available for {color}. Skipping anomaly detection.")
                    sliding_window[color].clear()

    finally:
        video_manager.release()
        video_manager.close_window()

def main():
    """Main function for live motion tracking."""
    config = load_json(CONFIG_PATH)
    training_coordinates = load_json(COORDINATES_PATH)
    cycle_data = load_json(CYCLE_PATH)

    # video_manager = VideoManager(source="videos/live_with_anomaly.mp4")
    video_manager = VideoManager(0)
    validate_configuration(config, video_manager)

    # Extract cycle_window for each color
    colors = config.get("colors", [])  # Dynamically fetch colors from config
    cycle_window = {color: cycle_data[color]["cycle_window"] for color in colors}
    avg_cycle = {color: cycle_data[color]["avg_cycle"] for color in colors}

    # Load PCA models for each color
    pca_models = {}
    for color in colors:
        pca_model_dir = f"pca/{color}"
        x_pca_path = os.path.join(pca_model_dir, "x_pca.pkl")
        y_pca_path = os.path.join(pca_model_dir, "y_pca.pkl")

        if os.path.exists(x_pca_path) and os.path.exists(y_pca_path):
            with open(x_pca_path, 'rb') as x_file, open(y_pca_path, 'rb') as y_file:
                x_pca = pickle.load(x_file)
                y_pca = pickle.load(y_file)
                pca_models[color] = (x_pca, y_pca)
                print(f"Loaded PCA models for {color}.")
        else:
            print(f"PCA model files not found for {color}: {x_pca_path} or {y_pca_path}")

    run_live_detection(video_manager, config, training_coordinates, pca_models, cycle_window, avg_cycle)

if __name__ == "__main__":
    main()
