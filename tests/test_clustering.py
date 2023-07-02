import pytest
import numpy as np
#from ruptures.detection import KMeansDetector
from ruptures.detection.clustering import KMeansDetector


def test_fit():
    # Test the fit method
    signal = np.random.rand(100, 1)
    kmd = KMeansDetector()
    kmd.fit(signal)
    assert kmd.signal is not None

def test_predict():
    # Test the predict method
    signal = np.random.rand(100, 1)
    kmd = KMeansDetector()
    kmd.fit(signal)
    bkps = kmd.predict()
    assert isinstance(bkps, np.ndarray)

def test_fit_predict():
    # Test fitting and predicting in one step
    signal = np.random.rand(100, 1)
    kmd = KMeansDetector()
    bkps = kmd.fit_predict(signal)
    assert isinstance(bkps, np.ndarray)

def test_fit_optimal_zero():
    # Test the _fit_optimal_zero method
    signal = np.random.rand(100, 1)
    kmd = KMeansDetector()
    kmd.fit(signal)
    kmd._fit_optimal_zero()
    assert kmd.cluster_centers_ is not None
    assert kmd.labels_ is not None
