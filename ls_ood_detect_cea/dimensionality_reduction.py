from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pacmap
from sklearn.decomposition import PCA


def apply_pca_ds(train_samples: np.ndarray,
                 test_samples: np.ndarray,
                 nro_components: int = 16,
                 svd_solver: str = 'randomized',
                 whiten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies PCA dimensionality reduction to a dataset

    :param train_samples: PCA train and fit samples
    :type train_samples: np.ndarray
    :param test_samples: PCA test samples
    :type test_samples: np.ndarray
    :param nro_components: PCA components number
    :type nro_components: int
    :param svd_solver: PCA SVD solver
    :type svd_solver: str
    :param whiten: PCA whiten value
    :type whiten: bool
    :return: Train and Test samples with reduced dimensionality
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    pca_dim_red = PCA(n_components=nro_components, svd_solver=svd_solver, whiten=whiten)
    train_ds = pca_dim_red.fit_transform(train_samples)
    test_ds = pca_dim_red.transform(test_samples)
    return train_ds, test_ds


def plot_samples_pacmap(samples_ind: np.ndarray,
                        samples_ood: np.ndarray,
                        neighbors: int = 25,
                        title: str = "Plot Title") -> None:
    """
    In-Distribution vs Out-of-Distribution Data Projection 2D Plot using PaCMAP algorithm.

    :param samples_ind: In-Distribution (InD) samples numpy array
    :type samples_ind: np.ndarray
    :param samples_ood: Out-of-Distribution (OoD) samples numpy array
    :type samples_ood: np.ndarray
    :param neighbors: Number of nearest-neighbors considered for the PaCMAP algorithm
    :type neighbors: int
    :param title: Plot tile
    :type title:  str
    :return:
    :rtype: None
    """
    samples_concat = np.concatenate((samples_ind, samples_ood))
    label_normal = np.zeros((samples_ind.shape[0], 1))
    label_anomaly = np.ones((samples_ood.shape[0], 1))
    labels = np.concatenate((label_normal, label_anomaly))
    print(samples_concat.shape)
    print(labels.shape)
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2.0)
    samples_transformed = embedding.fit_transform(samples_concat, init="pca")
    print(samples_transformed.shape)

    # visualize the embedding
    # ToDo: Add Axis Names and plot legend
    scatter = plt.scatter(samples_transformed[:, 0], samples_transformed[:, 1], cmap="brg", c=labels, s=1.5)
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=["In-Distribution", "Out-of-Distribution"])
    plt.show()
