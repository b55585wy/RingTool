import pytest
import numpy as np
import pandas as pd
import torch
from utils.accel_features import extract_accel_features, extract_accel_features_cuda


@pytest.fixture
def sample_data():
    """Generate sample accelerometer data for testing."""
    # Sample data with 100 points
    n = 100
    t = np.linspace(0, 1, n)  # 1 second of data at 100Hz
    
    # Create sine waves with different frequencies for x, y, z
    x = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
    y = np.sin(2 * np.pi * 3 * t)  # 3 Hz sine wave
    z = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    
    return x, y, z


def test_extract_accel_features(sample_data):
    """Test the pandas/numpy implementation of feature extraction."""
    x, y, z = sample_data
    
    # Call the function
    result = extract_accel_features(x, y, z, dt=0.01, window_size=10)
    
    # Check basic properties
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(x)
    
    # Check that all expected columns exist
    expected_columns = ['x', 'y', 'z', 'magnitude', 'sma', 'rms', 'jerk', 
                        'pitch', 'roll', 'pitch_deg', 'roll_deg']
    for col in expected_columns:
        assert col in result.columns
    
    # Test a few calculations
    # Magnitude = sqrt(x^2 + y^2 + z^2)
    expected_magnitude = np.sqrt(x**2 + y**2 + z**2)
    np.testing.assert_allclose(result['magnitude'].values, expected_magnitude, rtol=1e-5)
    
    # Check if jerk is calculated correctly for a few points
    expected_jerk = np.zeros_like(expected_magnitude)
    expected_jerk[1:] = (expected_magnitude[1:] - expected_magnitude[:-1]) / 0.01
    np.testing.assert_allclose(result['jerk'].values[1:], expected_jerk[1:], rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_extract_accel_features_cuda(sample_data):
    """Test the CUDA implementation of feature extraction."""
    x, y, z = sample_data
    
    # Call the function
    result = extract_accel_features_cuda(x, y, z, dt=0.01, window_size=10)
    
    # Check that it returns a dictionary
    assert isinstance(result, dict)
    
    # Check that all expected keys exist
    expected_keys = ['x', 'y', 'z', 'magnitude', 'sma', 'rms', 'jerk', 
                     'pitch', 'roll', 'pitch_deg', 'roll_deg']
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], torch.Tensor)
        assert result[key].shape[0] == len(x)
    
    # Test a few calculations (converting to CPU for comparison)
    # Magnitude = sqrt(x^2 + y^2 + z^2)
    expected_magnitude = np.sqrt(x**2 + y**2 + z**2)
    np.testing.assert_allclose(
        result['magnitude'].cpu().numpy(),
        expected_magnitude,
        rtol=1e-4
    )
    
    # Check if jerk is calculated correctly for a few points
    expected_jerk = np.zeros_like(expected_magnitude)
    expected_jerk[1:] = (expected_magnitude[1:] - expected_magnitude[:-1]) / 0.01
    np.testing.assert_allclose(
        result['jerk'].cpu().numpy()[1:], 
        expected_jerk[1:], 
        rtol=1e-4,
        atol=1e-5  # Add absolute tolerance to handle small differences
    )


def test_extract_accel_features_cuda_cpu_fallback(sample_data):
    """Test that the CUDA implementation works on CPU."""
    x, y, z = sample_data
    
    # Call the function with CPU device
    result = extract_accel_features_cuda(x, y, z, dt=0.01, window_size=10, device='cpu')
    
    # Check basic properties
    assert isinstance(result, dict)
    for key in result:
        assert result[key].device.type == 'cpu'
    
    # Test a calculation
    expected_magnitude = np.sqrt(x**2 + y**2 + z**2)
    np.testing.assert_allclose(
        result['magnitude'].numpy(), 
        expected_magnitude, 
        rtol=1e-5
    )