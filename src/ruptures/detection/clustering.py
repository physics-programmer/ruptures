from sklearn.cluster import KMeans
from ruptures.base import BaseEstimator
import numpy as np

class KMeansDetector(BaseEstimator):
    """
    Breakpoints detection with the KMeans clustering algorithm.

    Args:
        None

    Attributes:
        signal (numpy array): signal on which to perform change point detection.
        cluster_centers_ (numpy array): coordinates of cluster centers.
        labels_ (numpy array): Labels of each point.
    """

    def __init__(self):
        self.signal = None
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, signal):
        """
        Fit the model using the input signal.

        Args:
            signal (numpy array): signal to use for model fitting.
        """
        self.signal = signal
        return self

    def predict(self, n_bkps=None):
        """
        Predict breakpoints using the fitted model.

        Args:
            n_bkps (int, optional): number of breakpoints to predict. If not 
            specified, the optimal number is used.

        Returns:
            numpy array: breakpoints.
        """
        if n_bkps is None:
            self._fit_optimal_zero()
        else:
            kmeans = KMeans(n_clusters=n_bkps+1, random_state=0)
            self.labels_ = kmeans.fit_predict(self.signal)
            self.cluster_centers_ = kmeans.cluster_centers_
        return np.argwhere(np.diff(self.labels_)!=0).reshape(-1) + 1

    def fit_predict(self, signal, n_bkps=None):
        """
        Fit to data, then predict.

        Args:
            signal (numpy array): signal to use for model fitting.
            n_bkps (int, optional): number of breakpoints to predict. If not 
            specified, the optimal number is used.

        Returns:
            numpy array: breakpoints.
        """
        self.fit(signal)
        return self.predict(n_bkps)

    def _fit_optimal_zero(self):
        """
        Determine the optimal number of clusters by minimizing the sum of the 
        signal values in the cluster closest to zero.
        """
        kmeans = KMeans(n_clusters=2, random_state=0)
        current_error = np.inf
        while True:
            self.labels_ = kmeans.fit_predict(self.signal)
            self.cluster_centers_ = kmeans.cluster_centers_
            zero_cluster_index = np.argmin(self.cluster_centers_)
            zero_cluster_error = np.sum(
                (self.signal[self.labels_==zero_cluster_index]
                - self.cluster_centers_[zero_cluster_index])**2
            )
            if zero_cluster_error >= current_error:
                break
            current_error = zero_cluster_error
            kmeans.n_clusters += 1
