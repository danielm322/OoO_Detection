import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
from .dimensionality_reduction import apply_pca_ds
from .dimensionality_reduction import apply_pca_ds_split
from warnings import warn


def build_ood_detection_ds(
    ind_valid_data: np.ndarray,
    ood_valid_data: np.ndarray,
    ind_test_data: np.ndarray,
    ood_test_data: np.ndarray,
    apply_pca: bool = True,
    pca_nro_comp: int = 16,
    pca_svd_solver: str = "randomized",
    pca_whiten: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Adds labels of zeros and ones to some data. Optionally applies PCA to them. Deprecated method.

    Args:
        ind_valid_data: InD validation set data
        ood_valid_data: OoD validation set data
        ind_test_data: InD test set data
        ood_test_data: OoD test set data
        apply_pca: Optionally apply PCA, defaults to True
        pca_nro_comp: Number of PCA components, defaults to 16
        pca_svd_solver: PCA solver, defaults to 'randomized'
        pca_whiten: Whiten PCA, defaults to True

    Returns:
        train_ds, labels_train_ds, test_ds, labels_test_ds, pca_dim_red
    """
    warn(
        "This method is deprecated. Is not guaranteed to work with the rest of the library",
        DeprecationWarning,
        stacklevel=2,
    )
    pca_dim_red = None
    # Samples
    train_ds = np.vstack((ind_valid_data, ood_valid_data))
    test_ds = np.vstack((ind_test_data, ood_test_data))

    if apply_pca:
        train_ds, test_ds, pca_dim_red = apply_pca_ds(
            train_ds, test_ds, pca_nro_comp, pca_svd_solver, pca_whiten
        )

    # labels
    label_train_ind_ds = np.ones(
        (ind_valid_data.shape[0], 1)
    )  # 1: In-Distribution is the positive class
    label_train_ood_ds = np.zeros(
        (ood_valid_data.shape[0], 1)
    )  # 0: Out-of-Distribution/Anomaly is the negative class
    labels_train_ds = np.vstack((label_train_ind_ds, label_train_ood_ds))
    labels_train_ds = labels_train_ds.astype("int32")
    labels_train_ds = np.squeeze(labels_train_ds)

    label_test_ind_ds = np.ones(
        (ind_test_data.shape[0], 1)
    )  # 1: In-Distribution is the positive class
    label_test_ood_ds = np.zeros(
        (ood_test_data.shape[0], 1)
    )  # 0: Out-of-Distribution/Anomaly is the negative class
    labels_test_ds = np.vstack((label_test_ind_ds, label_test_ood_ds))
    labels_test_ds = labels_test_ds.astype("int32")
    labels_test_ds = np.squeeze(labels_test_ds)

    print("Train Dataset Samples shape: ", train_ds.shape)
    print("Train Dataset Labels shape: ", labels_train_ds.shape)

    print("Test Dataset Samples shape: ", test_ds.shape)
    print("Test Dataset Labels shape: ", labels_test_ds.shape)

    return train_ds, labels_train_ds, test_ds, labels_test_ds, pca_dim_red


def build_ood_detection_train_split(
    ind_data: np.ndarray,
    ood_data: np.ndarray,
    apply_pca: bool = True,
    pca_nro_comp: int = 16,
    pca_svd_solver: str = "randomized",
    pca_whiten: bool = True,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Adds labels of zeros and ones to some data. Optionally applies PCA to them. Deprecated method.

    Args:
        ind_data: InD data
        ood_data: OoD data
        apply_pca: Optionally apply PCA, defaults to True
        pca_nro_comp: Number of PCA components, defaults to 16
        pca_svd_solver: PCA solver, defaults to 'randomized'
        pca_whiten: Whiten PCA, defaults to True

    Returns:
        samples, labels, PCA estimator
    """
    warn(
        "This method is deprecated. "
        "Is not guaranteed to work with the rest of the library. "
        "Use the apply_pca_ds_split function instead",
        DeprecationWarning,
        stacklevel=2,
    )
    pca_dim_red = None
    # Samples
    samples = np.vstack((ind_data, ood_data))

    if apply_pca:
        samples, pca_dim_red = apply_pca_ds_split(samples, pca_nro_comp, pca_svd_solver, pca_whiten)

    # labels:
    label_ind_ds = np.ones((ind_data.shape[0], 1))  # 1: In-Distribution is the positive class
    label_ood_ds = np.zeros(
        (ood_data.shape[0], 1)
    )  # 0: Out-of-Distribution/Anomaly is the negative class

    labels = np.vstack((label_ind_ds, label_ood_ds))
    labels = labels.astype("int32")
    labels = np.squeeze(labels)

    print("Dataset Samples shape: ", samples.shape)
    print("Dataset Labels shape: ", labels.shape)

    return samples, labels, pca_dim_red
