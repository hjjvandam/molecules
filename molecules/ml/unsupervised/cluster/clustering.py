import numpy as np
from sklearn.cluster import DBSCAN, OPTICS


def dbscan_clustering(X, eps, min_samples, outlier_cutoff=10):
    """
    Optimizes eps parameter with outlier_cutoff and returns the
    optimal eps and associated outliers.

    Parameters
    ----------
    X : np.ndarray
        Data array to run DBSCAN on

    eps : float
        eps in the DBSCAN algorithm

    min_samples : int
        min_samples in the DBSCAN algorithm

    outlier_cutoff : int
        Keep updating eps while the number of outliers
        is greater than outlier_cutoff

    Return
    ------
    opt_eps : float
        optimal eps value given the X data

    outliers : np.ndarray
        opt_eps associated array of outlier indices in the X data

    labels : np.ndarray
        all cluster labels
    """

    opt_eps = eps
    # Search for optimal eps for DBSCAN 
    while True:
        # Run DBSCAN clustering on the data
        db = DBSCAN(eps=opt_eps, min_samples=min_samples).fit(X)
        # Array of outlier indices in latent space
        outlier_inds = np.flatnonzero(db.labels_ == -1)

        # If the number of outliers is greater than 150, update eps.
        if len(outlier_inds) > outlier_cutoff:
            opt_eps += 0.05
        else: 
            return opt_eps, outlier_inds, db.labels_


def optics_clustering(X, min_samples):
    """
    Performs default OPTICS clustering and returns associated 
    outlier indices.

    Parameters
    ----------
    X : np.ndarray
        Data array to run OPTICS on

    min_samples : int
        min_samples in the OPTICS algorithm

    Return
    ------
    opt_eps : float
        optimal eps value given the X data

    outliers : np.ndarray
        opt_eps associated array of outlier indices in the X data

    labels : np.ndarray
        all cluster labels
    """

    # Run OPTICS clustering on the data
    clustering = OPTICS(min_samples=min_samples).fit(X)
    
    # Array of outlier indices in X data
    outlier_inds = np.flatnonzero(clustering.labels_ == -1)
    
    return outlier_inds, clustering.labels_
