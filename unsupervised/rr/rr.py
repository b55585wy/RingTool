import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, butter, filtfilt

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """Apply a bandpass filter to the data."""
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_rr(y, fs=100, min=6, max=30, method = 'fft'):
    """
    Calculate heart rate using the fft or peak detection
    y: 0.1-0.5 Hz filtered PPG signal, only support 1D array
    fs: sampling frequency
    min: minimum heart rate
    max: maximum heart rate
    method: 'fft' or 'peak'
    """
    if y.ndim != 1:
        raise ValueError("Input signal y must be a 1D array.")

    if method == 'fft':
        p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=np.min((len(y)-1, 512)))
        fft_rr = p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
        return fft_rr
    elif method == 'peak':
        ppg_peaks, _ = find_peaks(y)
        peak_rr = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
        return peak_rr
    else:
        raise ValueError("Invalid method. Choose 'fft' or 'peak'.")
