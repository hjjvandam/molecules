import pytest
import numpy as np
from molecules.ml.unsupervised.cluster import (dbscan_clustering, 
                                               optics_clustering)


class TestClustering:

    @classmethod
    def reset_samples(self):
        samples = np.random.randn(10, 2)

        cluster1 = 1.0 * samples + 4
        cluster2 = 1.2 * samples + 10
        cluster3 = 1.3 * samples - 5
        self.outliers = np.array([[100, 100], [400, 500], [-400, 90], [300, 80]])

        self.X = np.concatenate((cluster1, cluster2, cluster3, self.outliers))
        np.random.shuffle(self.X)

    @classmethod
    def setup_class(self):
        self.reset_samples()

    def test_dbscan_clustering(self):
        eps = 0.2
        min_samples = 10

        # Attempts to correctly identify outliers with 10 different data sets
        for attempt in range(10):
            opt_eps, outlier_inds, labels = dbscan_clustering(self.X, eps, min_samples)

            if np.setdiff1d(self.outliers, self.X[outlier_inds]).size == 0:
                break
            self.reset_samples()

        else:
            assert False

    def test_optics_clustering(self):
        min_samples = 5

        # Attempts to correctly identify outliers with 10 different data sets
        for attempt in range(10):
            outlier_inds, labels = optics_clustering(self.X, min_samples)

            if np.setdiff1d(self.outliers, self.X[outlier_inds]).size == 0:
                break
            self.reset_samples()

        else:
            assert False

    @classmethod
    def teardown_class(self):
        pass