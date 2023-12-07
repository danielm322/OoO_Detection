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
    assert isinstance(hz_detector, DetectorKDE)
    scores = hz_detector.get_density_scores(samples)
    return scores


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian classifier for OoD Detection based on KDE. Taken from:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html.
    Deprecated method

    Args:
        bandwidth: The kernel bandwidth within each class
        kernel: the kernel name, passed to KernelDensity
    """

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        """
        Bayesian classifier for OoD Detection based on KDE. Taken from:
        https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html.
        Deprecated method

        Args:
            bandwidth: The kernel bandwidth within each class
            kernel: the kernel name, passed to KernelDensity
        """
        warn(
            "This method is deprecated. " "Is not guaranteed to work with the rest of the library",
            DeprecationWarning,
            stacklevel=2,
        )
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.classes_ = None
        self.models_ = None
        self.logpriors_ = None
        self.logprobs = None
        # self.result = None
        self.weighted_log_prob = None
        self.log_resp = None
        self.log_prob_norm = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))

        training_sets = [X[y == yi] for yi in self.classes_]

        self.models_ = [
            KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi)
            for Xi in training_sets
        ]

        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self

    def estimate_joint_log_prob(self, X):
        """Estimate the joint log-probabilities, log P(Z | Y) + log P(Y)."""
        self.logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        self.weighted_log_prob = self.logprobs + self.logpriors_  # numerator; priors act as weights
        return self.weighted_log_prob

    def predict_proba(self, X):
        "Evaluate the components' density for each sample"
        weighted_log_prob = self.estimate_joint_log_prob(X)
        self.log_prob_norm = logsumexp(weighted_log_prob, axis=1)

        with np.errstate(under="ignore"):
            # ignore underflow
            self.log_resp = weighted_log_prob - self.log_prob_norm[:, np.newaxis]

        return np.exp(self.log_resp)

    def predict_label(self, X):
        # return self.estimate_joint_log_prob(X).argmax(axis=1)
        return self.classes_[self.classes_[np.argmax(self.predict_proba(X), 1)]]

    def predict_prob(self, X):
        self.logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(self.logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def pred_prob(self, X):
        self.logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        log_result = self.logprobs + self.logpriors_
        log_prob_norm = logsumexp(log_result, axis=1)

        log_resp = log_result - log_prob_norm[:, np.newaxis]

        return np.exp(log_resp)

    def predict(self, X):
        # return self.classes_[np.argmax(self.predict_prob(X), 1)]
        return self.classes_[np.argmax(self.pred_prob(X), 1)]
