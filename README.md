# Computer Vision Anomaly Detection System

This project tracks a colored marker dot in video (like a dot on an arm/hand) and tries to tell when a repeating motion stops looking “normal”. It’s basically OpenCV tracking + some light signal processing on the tracked (x, y) coordinates.

## What it does (high level)
1. Tracks a colored dot in video (webcam or prerecorded video) and logs its (x, y) position over time.
2. Learns what “normal” looks like for a repeating motion:
	- estimates the cycle length (how long one rep is)
	- slices the tracking signal into cycles
	- computes FFT magnitude features per cycle
	- trains PCA models on the normalized FFT features
	- saves an average cycle template
3. Runs live anomaly detection by comparing the current window to the learned baseline:
	- cross-correlates to align the live window to the average cycle
	- computes FFT features and PCA reconstruction error
	- flags deviations (e.g., low correlation / high reconstruction error)

## Quickstart
1) Create the conda env:
- `conda env create -f environment.yml`
- `conda activate cv_fft_project`

2) Capture “normal” motion (writes `data/config.json` + `data/coordinates.json`):
- `python train.py`

3) Build the baseline model (generates `data/cycle.json`, `data/pca.json`, and PCA pickles under `pca/`):
- `python process_fft.py`

4) Run live detection:
- `python live.py`

Press `q` to quit the OpenCV window.

## Theory
- Track the dot each frame (HSV threshold -> biggest contour -> dot center).
- Turn dot positions into time series.
- Estimate the motion’s cycle length, average a “typical” cycle, and learn frequency patterns with FFT + PCA.
- In live mode, align the current window to the average cycle (cross-correlation) and flag deviations.

## Notes
- The config supports multiple colors, but `tracker_utils.py` currently defines HSV ranges for red, but it is possible for all color ranges provided there is contrast with the background.
- HSV thresholding is used here because it’s simple, fast, and works well when you have a high-contrast marker (no labeling/training step).
- Live mode checks FPS + resolution against what was saved during training.
- Lighting and marker contrast matter a lot for stability.

Why did I use signal processing instead of a CNN?
- Lower latency/compute overhead for real-time use (a CNN is overkill for “track a colored dot + compare motion patterns”).
- Lightweight + explainable baseline: learn the repetition pattern from the tracked (x, y) signal (cycle length, correlation, FFT features) and flag when it drifts.
- Performs well for a mechanical actuator undergoing a stress test, where the motion is repetitive and deviations are meaningful. I tested this with a industrial stress test setup and it was able to detect when a actuator fails near 100% of the time.

