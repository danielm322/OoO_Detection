# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez
#    Daniel Montoya

"""Module containing the KDE Detectors"""
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from warnings import warn


class DetectorKDE:
    """
    Instantiates a Kernel Density Estimation Estimator. See
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html for
    more details

    Args:
        train_embeddings: Samples to train the estimator
        save_path: Optional path to save the estimator
        kernel: Kernel. Default='gaussian'
        bandwidth: Bandwidth of the estimator.
    """

    def __init__(self, train_embeddings, save_path=None, kernel="gaussian", bandwidth=1.0) -> None:
        """
        Instantiates a Kernel Density Estimation Estimator. See
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html for
        more details

        Args:
            train_embeddings: Samples to train the estimator
            save_path: Optional path to save the estimator
            kernel: Kernel. Default='gaussian'
            bandwidth: Bandwidth of the estimator.
        """
        assert isinstance(train_embeddings, np.ndarray), "train_embeddings must be a numpy array"
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.train_embeddings = train_embeddings
        self.save_path = save_path
        self.density = self.density_fit()

    def density_fit(self):
        """
        Fit the KDE Estimator
        """
        density = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(
            self.train_embeddings
        )
        return density

    def get_density_scores(self, test_embeddings):
        """
        Transforms the scores from a second distribution while normalizing the scores

        Args:
            test_embeddings: The new samples to get the density scores

        Returns:
            Density scores
        """
        return self.density.score_samples(test_embeddings)


def get_hz_scores(hz_detector: DetectorKDE, samples: np.ndarray):
    """
    Performs inference with an already trained KDE detector

    Args:
        hz_detector: The trained estimator
        samples: The new samples to be scored

    Returns:
        The density scores
    """
    warn(
        "This method will be deprecated. ",
        DeprecationWarning,
        stacklevel=2,
    )
    assert isinstance(hz_detector, DetectorKDE)
    scores = hz_detector.get_density_scores(samples)
    return scores
