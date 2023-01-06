import numpy as np
import array
from dtaidistance import dtw
import sys
from scipy.stats import mode
from scipy.spatial.distance import squareform
from scipy.spatial.distance import euclidean
# from dtw import *
from fastdtw import fastdtw

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the time series array. By setting subsample_step = 2,
        the time series length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, label):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x: [{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]
        {
            'accelX': [],
            'accelY': [],
            'accelZ': []
        }
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        self.x = x
        self.label = np.array(label)

    def dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """
        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                           min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    def dist_matrix(self, x, y, rss = lambda x,y,z: np.sqrt(x**2+y**2+z**2)):
        """Computes the M x N distance matrix between the training
        dataset(x) and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x: [{ACTIVITY 1}, {ACTIVITY 2}, {ACTIVITY 3}, ...]
        {
            'accelX': [],
            'accelY': [],
            'accelZ': []
        }
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if (np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            # p = ProgressBar(shape(dm)[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self.dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    # p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            # dm_size = x_s[0] * y_s[0]

            # p = ProgressBar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    x_accelX = x[i]["accelX"]
                    x_accelY = x[i]["accelY"]
                    x_accelZ = x[i]["accelZ"]
                    y_accelX = y[j]["accelX"]
                    y_accelY = y[j]["accelY"]
                    y_accelZ = y[j]["accelZ"]
                    # 3D modification
                    dm[i, j] =rss(self.dtw_distance(x_accelX, y_accelX),
                                  self.dtw_distance(x_accelY, y_accelY),
                                  self.dtw_distance(x_accelZ, y_accelZ))
                    # dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                    #                               y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    # p.animate(dm_count)

            return dm

    def dist_matrix_py(self, x, y, rss = lambda x,y,z: np.sqrt(x**2+y**2+z**2)):
        # Compute the distance matrix
        # dm_count = 0
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.zeros((x_s[0], y_s[0]))
        # dm_size = x_s[0] * y_s[0]

        # p = ProgressBar(dm_size)

        for i in range(0, x_s[0]):
            for j in range(0, y_s[0]):
                x_accelX = [round(n, 2) for n in x[i]["accelX"]]
                x_accelY = [round(n, 2) for n in x[i]["accelY"]]
                x_accelZ = [round(n, 2) for n in x[i]["accelZ"]]
                y_accelX = [round(n, 2) for n in y[j]["accelX"]]
                y_accelY = [round(n, 2) for n in y[j]["accelY"]]
                y_accelZ = [round(n, 2) for n in y[j]["accelZ"]]
                # 3D modification

                dm[i, j] = rss(dtw(x_accelX, y_accelX, distance_only=True).normalizedDistance,
                               dtw(x_accelY, y_accelY, distance_only=True).normalizedDistance,
                               dtw(x_accelZ, y_accelZ, distance_only=True).normalizedDistance)
                # dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                #                               y[j, ::self.subsample_step])
                # Update progress bar
                # dm_count += 1
                # p.animate(dm_count)

        return dm

    def dist_matrix_py_fast(self, x, y, rss = lambda x,y,z: np.sqrt(x**2+y**2+z**2)):
        # Compute the distance matrix
        # dm_count = 0
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.zeros((x_s[0], y_s[0]))
        # dm_size = x_s[0] * y_s[0]

        # p = ProgressBar(dm_size)

        for i in range(0, x_s[0]):
            for j in range(0, y_s[0]):
                # x_accelX = array.array('d', [round(n, 2) for n in x[i]["accelX"]])
                # x_accelY = array.array('d', [round(n, 2) for n in x[i]["accelY"]])
                # x_accelZ = array.array('d', [round(n, 2) for n in x[i]["accelZ"]])
                # y_accelX = array.array('d', [round(n, 2) for n in y[j]["accelX"]])
                # y_accelY = array.array('d', [round(n, 2) for n in y[j]["accelY"]])
                # y_accelZ = array.array('d', [round(n, 2) for n in y[j]["accelZ"]])
                x_accelX = array.array('d', x[i]["accelX"])
                x_accelY = array.array('d', x[i]["accelY"])
                x_accelZ = array.array('d', x[i]["accelZ"])
                y_accelX = array.array('d', y[j]["accelX"])
                y_accelY = array.array('d', y[j]["accelY"])
                y_accelZ = array.array('d', y[j]["accelZ"])
                # 3D modification

                # dm[i, j] = rss(fastdtw(x_accelX, y_accelX, dist=euclidean),
                #                fastdtw(x_accelY, y_accelY, dist=euclidean),
                #                fastdtw(x_accelZ, y_accelZ, dist=euclidean))
                dm[i, j] = rss(dtw.distance_fast(x_accelX, y_accelX, use_pruning=True),
                               dtw.distance_fast(x_accelY, y_accelY, use_pruning=True),
                               dtw.distance_fast(x_accelZ, y_accelZ, use_pruning=True))
                # dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                #                               y[j, ::self.subsample_step])
                # Update progress bar
                # dm_count += 1
                # p.animate(dm_count)

        return dm

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
            x : array of shape [n_samples, n_timepoints]
                Array containing the testing data set to be classified

        Returns
        -------
            2 arrays representing:
                (1) the predicted class labels
                (2) the knn label count probability
        """
        # x: testing data
        # self.x: training
        # dm = self.dist_matrix(x, self.x)
        # dm = self.dist_matrix_py(x, self.x)
        dm = self.dist_matrix_py_fast(x, self.x)
        # print(f'distance matrix: {dm}')
        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.label[knn_idx]

        # Model Label
        # scipy.stats.mode: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mode.html
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors
        # modified version
        for i in range(0, mode_data[1].shape[0]):
            if mode_data[1][i, 0] == 1:
                mode_label[i, 0] = knn_labels[i, 0]

        return mode_label.ravel(), mode_proba.ravel()

    def evaluate(self):
        pass