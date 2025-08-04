import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from utils import load_json
from sklearn.decomposition import PCA
import json
import os
import pickle  # Importing pickle for saving PCA model

def estimate_cycle_length(series):
    """Estimate cycle length using autocorrelation with a minimum lag of 20."""
    # Remove the mean to focus on variations
    series = series - np.mean(series)

    # Compute autocorrelation
    autocorr = correlate(series, series, mode='full')
    mid = len(autocorr) // 2
    autocorr = autocorr[mid:]  # Use only the second half

    # Find the first peak in the autocorrelation after lag 20
    cycle_length = np.argmax(autocorr[20:]) + 20

    return cycle_length

def slice_signal(series, cycle_length):
    """Slice signal into windows of one cycle length."""
    return [series[i:i + cycle_length] for i in range(0, len(series), cycle_length) if len(series[i:i + cycle_length]) == cycle_length]

def save_json(data, filename):
    """Save data to a JSON file, converting numpy types to Python types."""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=convert_numpy)

def main(debug=False):
    """Process x_series and y_series: estimate cycle length, slice into windows, apply FFT, normalize, train PCA, and plot results."""
    data = load_json("data/coordinates.json")
    try:
        config = load_json("data/config.json")
    except FileNotFoundError:
        config = {}

    pca_parameters = {}

    for color, series in data.items():
        # Step 1: Estimate cycle length
        x_cycle_length = estimate_cycle_length(series['x_series'])
        y_cycle_length = estimate_cycle_length(series['y_series'])
        print(f"Cycle length for {color} (X): {x_cycle_length}")
        print(f"Cycle length for {color} (Y): {y_cycle_length}")

        # Step 2: Slice into windows
        x_windows = slice_signal(series['x_series'], x_cycle_length)
        y_windows = slice_signal(series['y_series'], y_cycle_length)

        # Step 3: Apply FFT to each window
        x_ffts = [np.abs(np.fft.rfft(w)) for w in x_windows]
        y_ffts = [np.abs(np.fft.rfft(w)) for w in y_windows]

        # Step 4: Normalize each FFT
        x_ffts_normed = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros_like(v) for v in x_ffts]
        y_ffts_normed = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros_like(v) for v in y_ffts]

        # Step 5: Train PCA models
        x_pca = PCA(n_components=0.95).fit(x_ffts_normed)
        y_pca = PCA(n_components=0.95).fit(y_ffts_normed)

        print(f"PCA components for {color} (X): {x_pca.components_}")
        print(f"PCA components for {color} (Y): {y_pca.components_}")

        # Save actual PCA parameters
        pca_parameters[color] = {
            "x_pca_mean": x_pca.mean_.tolist(),
            "x_pca_components": x_pca.components_.tolist(),
            "x_pca_explained_variance": x_pca.explained_variance_.tolist(),
            "y_pca_mean": y_pca.mean_.tolist(),
            "y_pca_components": y_pca.components_.tolist(),
            "y_pca_explained_variance": y_pca.explained_variance_.tolist()
        }

        # Save PCA models to individual pickle files for each color in a dedicated folder
        pca_dir = f"pca/{color}"
        os.makedirs(pca_dir, exist_ok=True)

        pickle.dump(x_pca, open(os.path.join(pca_dir, "x_pca.pkl"), "wb"))
        pickle.dump(y_pca, open(os.path.join(pca_dir, "y_pca.pkl"), "wb"))

        # Plot overlayed slices of x and y signals
        plt.figure(figsize=(10, 6))
        for x_slice in x_windows:
            plt.plot(x_slice, alpha=0.5, label=f"{color} X Slice")
        for y_slice in y_windows:
            plt.plot(y_slice, alpha=0.5, label=f"{color} Y Slice")
        plt.title(f"Overlayed Sliced Signals for {color}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate and plot the average of overlayed raw x and y overlapped slices
        avg_x_slice = np.mean(x_windows, axis=0)
        avg_y_slice = np.mean(y_windows, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(avg_x_slice, label=f"{color} Average X Slice", linewidth=2)
        plt.plot(avg_y_slice, label=f"{color} Average Y Slice", linewidth=2)
        plt.title(f"Average Overlayed Sliced Signals for {color}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

        if debug:
            # Overlay FFT plots
            plt.figure(figsize=(10, 6))
            for fft_x in x_ffts:
                plt.plot(fft_x, alpha=0.5, label=f"{color} X FFT")
            for fft_y in y_ffts:
                plt.plot(fft_y, alpha=0.5, label=f"{color} Y FFT")
            plt.title(f"Overlayed FFTs for {color}")
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Calculate actual average cycle of x and y time series values
        avg_x_cycle = np.mean(x_windows, axis=0).tolist()
        avg_y_cycle = np.mean(y_windows, axis=0).tolist()

        # Save cycle data to cycle.json with updated structure
        cycle_data = load_json("data/cycle.json") if os.path.exists("data/cycle.json") else {}
        cycle_data[color] = {
            "avg_cycle": {
                "x": avg_x_cycle,
                "y": avg_y_cycle
            },
            "cycle_window": {
                "x": x_cycle_length,
                "y": y_cycle_length
            }
        }
        save_json(cycle_data, "data/cycle.json")

    # Save actual PCA parameters to pca.json
    save_json(pca_parameters, "data/pca.json")

if __name__ == "__main__":
    main(debug=True)
