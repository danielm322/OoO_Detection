import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
from .dimensionality_reduction import apply_pca_ds
from .dimensionality_reduction import apply_pca_ds_split
from .dimensionality_reduction import apply_pca_transform


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

    :param ind_valid_data:
    :type ind_valid_data:
    :param ood_valid_data:
    :type ood_valid_data:
    :param ind_test_data:
    :type ind_test_data:
    :param ood_test_data:
    :type ood_test_data:
    :param apply_pca:
    :type apply_pca:
    :param pca_nro_comp:
    :type pca_nro_comp:
    :param pca_svd_solver:
    :type pca_svd_solver:
    :param pca_whiten:
    :type pca_whiten:
    :return:
    :rtype:
    """
    pca_dim_red = None
    # Samples
    train_ds = np.vstack((ind_valid_data, ood_valid_data))
    test_ds = np.vstack((ind_test_data, ood_test_data))

    if apply_pca:
        train_ds, test_ds, pca_dim_red = apply_pca_ds(train_ds, test_ds, pca_nro_comp, pca_svd_solver, pca_whiten)

    # labels
    label_train_ind_ds = np.ones((ind_valid_data.shape[0], 1))  # 1: In-Distribution is the positive class
    label_train_ood_ds = np.zeros((ood_valid_data.shape[0], 1))  # 0: Out-of-Distribution/Anomaly is the negative class
    labels_train_ds = np.vstack((label_train_ind_ds, label_train_ood_ds))
    labels_train_ds = labels_train_ds.astype("int32")
    labels_train_ds = np.squeeze(labels_train_ds)

    label_test_ind_ds = np.ones((ind_test_data.shape[0], 1))  # 1: In-Distribution is the positive class
    label_test_ood_ds = np.zeros((ood_test_data.shape[0], 1))  # 0: Out-of-Distribution/Anomaly is the negative class
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
    """_summary_

    :param ind_data: _description_
    :type ind_data: np.ndarray
    :param ood_data: _description_
    :type ood_data: np.ndarray
    :param pca_transform: _description_, defaults to None
    :type pca_transform: PCA, optional
    :param apply_pca: _description_, defaults to True
    :type apply_pca: bool, optional
    :param pca_nro_comp: _description_, defaults to 16
    :type pca_nro_comp: int, optional
    :param pca_svd_solver: _description_, defaults to 'randomized'
    :type pca_svd_solver: str, optional
    :param pca_whiten: _description_, defaults to True
    :type pca_whiten: bool, optional
    :return: _description_
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    pca_dim_red = None
    # Samples
    samples = np.vstack((ind_data, ood_data))

    if apply_pca:
        samples, pca_dim_red = apply_pca_ds_split(samples, pca_nro_comp, pca_svd_solver, pca_whiten)

    # labels:
    label_ind_ds = np.ones((ind_data.shape[0], 1))  # 1: In-Distribution is the positive class
    label_ood_ds = np.zeros((ood_data.shape[0], 1))  # 0: Out-of-Distribution/Anomaly is the negative class

    labels = np.vstack((label_ind_ds, label_ood_ds))
    labels = labels.astype("int32")
    labels = np.squeeze(labels)

    print("Dataset Samples shape: ", samples.shape)
    print("Dataset Labels shape: ", labels.shape)

    return samples, labels, pca_dim_red


def build_ood_detection_test_split(
    ind_data: np.ndarray, ood_data: np.ndarray, pca_transform: PCA = None
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    :param ind_data: _description_
    :type ind_data: np.ndarray
    :param ood_data: _description_
    :type ood_data: np.ndarray
    :param pca_transform: _description_, defaults to None
    :type pca_transform: PCA, optional
    :return: _description_
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    # Samples
    samples = np.vstack((ind_data, ood_data))

    if pca_transform is not None:
        samples = apply_pca_transform(samples, pca_transform)

    # labels:
    label_ind_ds = np.ones((ind_data.shape[0], 1))  # 1: In-Distribution is the positive class
    label_ood_ds = np.zeros((ood_data.shape[0], 1))  # 0: Out-of-Distribution/Anomaly is the negative class

    labels = np.vstack((label_ind_ds, label_ood_ds))
    labels = labels.astype("int32")
    labels = np.squeeze(labels)

    print("Dataset Samples shape: ", samples.shape)
    print("Dataset Labels shape: ", labels.shape)

    return samples, labels
