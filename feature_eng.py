import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks

def compute_magnitude_features(signal_samples):
    """
    Compute magnitude-related features for the given signal samples.

    Features:
    - Mean Amplitude: Average of the signal values.
    - Peak-to-Peak Amplitude: Difference between the max and min values.
    - Root Mean Square (RMS): Square root of the mean of squared signal values.

    Args:
        signal_samples (list or np.array): Input signal values.

    Returns:
        list: [mean_amplitude, peak_to_peak_amplitude, rms]
    """
    if len(signal_samples) > 1:
        mean_amplitude = np.mean(signal_samples)
        ptp_amplitude = np.ptp(signal_samples)
        rms = np.sqrt(np.mean(np.square(signal_samples)))
        return [mean_amplitude, ptp_amplitude, rms]
    else:
        return [0] * 3

def compute_integral_features(signal_samples):
    if len(signal_samples) > 1:
        integral = np.trapz(signal_samples) / len(signal_samples)
        squared_integral = np.trapz(np.square(signal_samples)) / len(signal_samples)
        return [integral, squared_integral]
    else:
        return [0] * 2

def compute_fft_features(sample, sampling_rate=100):
    if len(sample) > 1:
        fft = rfft(sample)
        power_spectrum = np.square(np.abs(fft))
        energy = np.mean(power_spectrum)
        power = np.sum(power_spectrum)
        return [energy, power]
    else:
        return [0] * 2

def denoise_signal(signal, sampling_rate=100):
    """
    Apply FFT, filter the signal based on the cutoff frequency, and return the inverse RFFT.

    Parameters:
    - signal: numpy array, the input time-domain signal
    - sampling_rate: float, the sampling rate of the signal in Hz

    Returns:
    - filtered_signal: numpy array, the filtered signal in the time domain
    """
    fft_coefficients = rfft(signal)
    magnitude = np.abs(fft_coefficients)
    threshold = np.mean(magnitude)
    significant_indices = magnitude > threshold

    filtered_fft = np.zeros_like(fft_coefficients)
    filtered_fft[significant_indices] = fft_coefficients[significant_indices]
    return irfft(filtered_fft, n=len(signal))

def extract_feature_set(signal_data):
    magnitude_features = compute_magnitude_features(signal_data)
    integral_features = compute_integral_features(signal_data)
    fft_features = compute_fft_features(signal_data)
    return magnitude_features + integral_features + fft_features

def find_true_peak(signal, prominence_range=(0.25, 1.5), width_range=(10, 60)):
    peaks, _ = find_peaks(signal, prominence=prominence_range, width=width_range)
    return peaks[np.argmax(signal[peaks])] if len(peaks) > 0 else np.argmax(signal)

def find_active_point(smoothed_signal, peak_index):
    derivatives = np.gradient(smoothed_signal)
    for i in range(peak_index, 0, -1):
        if derivatives[i - 1] <= 0.0:
            return i
    return 0

def find_decay_point(smoothed_signal, peak_index, decay_duration=20):
    derivatives = np.gradient(smoothed_signal)
    for j in range(peak_index, len(derivatives) - decay_duration):
        if all(derivatives[j: j + decay_duration] < 0):
            return j
    return peak_index

def filter_signal(smoothed_signal):
    peak_index = find_true_peak(smoothed_signal)
    point_A = find_active_point(smoothed_signal, peak_index)
    point_C = find_decay_point(smoothed_signal, peak_index)
    return smoothed_signal[point_A: point_C + 1]

def generate_features(df):
    sensor_feature_names = []
    for sensor in constants.SENSOR_COLS:
        for feature in ['AVG', 'PTP', 'RMS', 'INT', 'SQ_INT', 'ENERGY', 'POWER']:
            sensor_feature_names.append(f"{sensor}_{feature}")

    feature_vector = []
    for sensor_name in constants.SENSOR_COLS:
        smooth_signal = denoise_signal(signal=df[sensor_name].tolist())
        filtered_samples = filter_signal(smooth_signal)
        sensor_features = extract_feature_set(filtered_samples)
        feature_vector.extend(sensor_features)

    features = pd.DataFrame([feature_vector], columns=sensor_feature_names)
    return features
