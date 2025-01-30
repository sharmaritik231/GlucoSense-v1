import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks

def plot_voltage_signal(signal_array, heading: str):
    # Creates the voltage vs time graph!
    plt.figure(figsize=(6, 5))
    plt.plot(signal_array, 'b')
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")
    plt.title(label=f"BGL = {heading}")
    plt.show()


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
        length = len(sample)
        fft = rfft(sample)
        freq = rfftfreq(length, d= 1 / sampling_rate)

        # Power spectrum for positive frequencies
        power_spectrum = np.square(np.abs(fft))

        # Existing features
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
    # Perform the FFT and calculate the frequency bins
    N = len(signal)
    t = np.linspace(0, N/sampling_rate, N, endpoint=False)

    fft_coefficients = rfft(signal)
    frequencies = rfftfreq(N, d=1 / sampling_rate)

    magnitude = np.abs(fft_coefficients)
    threshold = np.mean(magnitude)
    significant_indices = magnitude > threshold

    filtered_fft = np.zeros_like(fft_coefficients)
    filtered_fft[significant_indices] = fft_coefficients[significant_indices]
    filtered_signal = irfft(filtered_fft, n=N)

    # Subdivide frequency bands and reconstruct individual components in time domain
    components = []
    for i in range(len(frequencies)):
        component_fft = np.zeros_like(fft_coefficients)
        component_fft[i] = fft_coefficients[i]  # Retain only one frequency component
        component = np.fft.irfft(component_fft, n=N)  # Inverse FFT to get time-domain signal
        components.append(component)

    return filtered_signal


def extract_feature_set(signal_data):
    comb_feature_set = []
    magnitude_features = compute_magnitude_features(signal_data)
    integral_features = compute_integral_features(signal_data)
    fft_features = compute_fft_features(signal_data)
    comb_feature_set += magnitude_features + integral_features + fft_features
    return comb_feature_set

def get_patient_singular_data_and_labels(patient_df):
    singular_features = []

    # For BGL Case & Reg case!
    for column_name in constants.TEST_SINGULAR_NAMES:
        singular_features.append(patient_df[column_name].unique()[0])

    return singular_features[:-1], singular_features[-1]


def find_true_peak(signal, prominence_range=(0.25, 1.5), width_range=(10, 60)):
    # Find all peaks with adjusted criteria
    peaks, properties = find_peaks(signal, prominence=prominence_range, width=width_range)

    # Get the true peak (the maximum value among valid peaks)
    if len(peaks) > 0:
        return peaks[np.argmax(signal[peaks])]
    else:
        return np.argmax(signal)


def find_active_point(smoothed_signal, peak_index):
    derivatives = np.gradient(smoothed_signal)
    point_A = 0
    # A: point of continuous rise up to the true peak!
    for i in range(peak_index, 0, -1):
        if derivatives[i - 1] <= 0.0:  # Look for where the rise starts
            point_A = i
            break
    return point_A


def find_decay_point(smoothed_signal, peak_index, decay_duration=20):
    derivatives = np.gradient(smoothed_signal)
    point_C = peak_index
    # The point after the true peak when the signal decreases for a specific duration!
    for j in range(peak_index, len(derivatives) - decay_duration):
        if all(derivatives[j: j + decay_duration] < 0):  # Continuous decay
            point_C = j
            break
    return point_C


def filter_signal(smoothed_signal):
    peak_index = find_true_peak(smoothed_signal)  # True Peak of the signal!
    point_A = find_active_point(smoothed_signal, peak_index)  # Point A (start of continuous rise)
    point_C = find_decay_point(smoothed_signal, peak_index)  # Point C (start of continuous decay)
    return smoothed_signal[point_A: point_C + 1]

def generate_features(df):
    magnitude_names = ['AVG', 'PTP', 'RMS']
    integral_names = ['INT', 'SQ_INT']
    fft_feature_names = ['ENERGY', 'POWER']
    sensor_feature_names = []

    # For each sensor iteration adding feature name!
    for n in range(len(constants.SENSOR_COLS)):
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in magnitude_names]
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in integral_names]
        sensor_feature_names += [f"{constants.SENSOR_COLS[n]}_{name}" for name in fft_feature_names]

    feature_vector = []
    for sensor_name in constants.SENSOR_COLS:
        smooth_signal = denoise_signal(signal=df[sensor_name].tolist())
        filtered_samples = filter_signal(smooth_signal)
        sensor_features = extract_feature_set(filtered_samples)
        feature_vector = feature_vector + sensor_features

    feature_vector = np.array([feature_vector])
    features = pd.DataFrame(data=feature_vector, columns=sensor_feature_names)
    return features