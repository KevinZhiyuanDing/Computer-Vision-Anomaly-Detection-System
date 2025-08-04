import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity
from utils import interpolate_nan_values

def process_fft(data_series, sampling_rate, window_size):
    """Process FFT for a given time series data."""
    # Convert input data to a NumPy array and handle missing values
    data_series = np.array(data_series, dtype=np.float32)
    data_series = interpolate_nan_values(data_series)

    # Remove the mean (DC component) to focus on frequency variations
    data_series = data_series - np.mean(data_series)

    # Compute the FFT and take the absolute value to get magnitudes
    fft_result = np.abs(fft(data_series))

    # Generate frequency bins for the FFT result
    frequencies = np.fft.rfftfreq(window_size, d=1 / sampling_rate)

    # Return the FFT frequencies and amplitudes
    return {
        "frequencies": frequencies,
        "amplitudes": fft_result[:len(frequencies)]
    }

def process_psd(data_series, sampling_rate, window_size):
    """Process Welch's PSD for a given time series data."""
    data_series = np.array(data_series, dtype=np.float32)
    frequencies, psd = welch(data_series, fs=sampling_rate, nperseg=window_size)
    return {
        "frequencies": frequencies,
        "psd": psd
    }

def normalize_fft_magnitude(fft_result):
    """Normalize FFT magnitude to unit length."""
    amplitudes = np.array(fft_result['amplitudes'], dtype=np.float32)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        return amplitudes / norm
    return amplitudes

def plot_fft(frequencies, fft_values, title="FFT Graph"):
    """Plot the FFT graph with max frequency and amplitude annotated."""
    plt.figure(1)  # Reuse the same figure
    plt.clf()  # Clear the current figure before plotting
    plt.plot(frequencies, fft_values, label="FFT Magnitude")
    
    # Annotate max frequency and amplitude
    max_idx = np.argmax(fft_values)
    max_freq = frequencies[max_idx]
    max_amp = fft_values[max_idx]
    plt.annotate(f"Max: {max_freq} Hz, {max_amp}",
                 xy=(max_freq, max_amp),
                 xytext=(max_freq + 0.1 * max_freq, max_amp + 0.1 * max_amp),
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=10, color='red')

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_psd(frequencies, psd_values, title="PSD Graph"):
    """Plot the PSD graph with max frequency and power annotated."""
    plt.figure(1)  # Reuse the same figure
    plt.clf()  # Clear the current figure before plotting
    plt.plot(frequencies, psd_values, label="PSD Power")

    # Annotate max frequency and power
    max_idx = np.argmax(psd_values)
    max_freq = frequencies[max_idx]
    max_power = psd_values[max_idx]
    plt.annotate(f"Max: {max_freq} Hz, {max_power}",
                 xy=(max_freq, max_power),
                 xytext=(max_freq + 0.1 * max_freq, max_power + 0.1 * max_power),
                 arrowprops=dict(facecolor='blue', arrowstyle='->'),
                 fontsize=10, color='blue')

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid(True)
    plt.legend()
    plt.show()

# Define the workspace root dynamically
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(__file__))

# Update paths to use absolute paths
TRAINING_DATA_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'fft_training_data.json')
VIDEOS_PATH = os.path.join(WORKSPACE_ROOT, 'videos', 'train.avi')

training_path = TRAINING_DATA_PATH
camera_index = VIDEOS_PATH
