from typing import Dict

import numpy as np
import pandas as pd
import torch


def extract_accel_features(x, y, z, dt=0.01, window_size=10) -> pd.DataFrame:
    """
    Extracts combined features from accelerometer data.

    Parameters:
    - x, y, z: Lists or numpy arrays of the same length
    - dt: Time delta between samples (in seconds), default 0.01 (100 Hz)
    - window_size: Rolling window size for SMA and RMS, default 10

    Returns:
    - DataFrame with original and derived features
    """
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # 1. Magnitude
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    # 2. SMA (Signal Magnitude Area)
    df['sma'] = (np.abs(df['x']) + np.abs(df['y']) + np.abs(df['z'])).rolling(window=window_size).mean()

    # 3. RMS (Root Mean Square)
    df['rms'] = df[['x', 'y', 'z']].apply(lambda row: np.sqrt(np.mean(row**2)), axis=1)

    # 4. Jerk (rate of change of magnitude)
    df['jerk'] = df['magnitude'].diff() / dt

    # 5. Orientation: Pitch & Roll (in radians and degrees)
    df['pitch'] = np.arctan2(df['x'], np.sqrt(df['y']**2 + df['z']**2))
    df['roll'] = np.arctan2(df['y'], df['z'])
    df['pitch_deg'] = np.degrees(df['pitch'])
    df['roll_deg'] = np.degrees(df['roll'])

    return df


def extract_accel_features_cuda(x, y, z, dt=0.01, window_size=10, device='cuda:0') -> Dict[str, torch.Tensor]:
    """
    Extracts combined features from accelerometer data using PyTorch with CUDA.

    Parameters:
    - x, y, z: Lists or numpy arrays of the same length
    - dt: Time interval between samples (seconds)
    - window_size: Rolling window size (no smoothing kernel applied here)
    - device: 'cuda' or 'cpu'

    Returns:
    - Dictionary of torch.Tensors with derived features
    """
    # Convert to torch tensors on GPU
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    z = torch.tensor(z, dtype=torch.float32, device=device)

    # 1. Magnitude
    magnitude = torch.sqrt(x**2 + y**2 + z**2)

    # 2. SMA: Rolling mean of |x| + |y| + |z|
    abs_sum = torch.abs(x) + torch.abs(y) + torch.abs(z)
    sma = torch.nn.functional.avg_pool1d(abs_sum.view(1, 1, -1), kernel_size=window_size, stride=1, padding=window_size//2)
    sma = sma.view(-1)
    # Ensure sma has the same length as original data
    if sma.shape[0] != x.shape[0]:
        sma = torch.nn.functional.pad(sma, (0, x.shape[0] - sma.shape[0])) if sma.shape[0] < x.shape[0] else sma[:x.shape[0]]

    # 3. RMS
    rms = torch.sqrt((x**2 + y**2 + z**2) / 3)

    # 4. Jerk
    jerk = torch.zeros_like(magnitude)
    jerk[1:] = (magnitude[1:] - magnitude[:-1]) / dt

    # 5. Orientation
    pitch = torch.atan2(x, torch.sqrt(y**2 + z**2))
    roll = torch.atan2(y, z)
    pitch_deg = pitch * (180.0 / torch.pi)
    roll_deg = roll * (180.0 / torch.pi)

    return {
        'x': x,
        'y': y,
        'z': z,
        'magnitude': magnitude,
        'sma': sma,
        'rms': rms,
        'jerk': jerk,
        'pitch': pitch,
        'roll': roll,
        'pitch_deg': pitch_deg,
        'roll_deg': roll_deg
    }
