# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Fabio Arnez

import numpy as np
from typing import Tuple
from .dimensionality_reduction import apply_pca_ds


def build_ood_detection_ds(ind_valid_data: np.ndarray,
                           ood_valid_data: np.ndarray,
                           ind_test_data: np.ndarray,
                           ood_test_data: np.ndarray,
                           apply_pca: bool = True,
                           pca_nro_comp: int = 16,
                           pca_svd_solver: str = 'randomized',
                           pca_whiten: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # Samples
    train_ds = np.vstack((ind_valid_data, ood_valid_data))
    test_ds = np.vstack((ind_test_data, ood_test_data))

    if apply_pca:
        train_ds, test_ds = apply_pca_ds(train_ds, test_ds, pca_nro_comp, pca_svd_solver, pca_whiten)

    # if dnn_architecture == "deeplabv3+":
    #     # Compute the components and projected faces
    #     pca_dlv3p_arch = PCA(n_components=16, svd_solver='randomized', whiten=True)
    #     train_ds = pca_dlv3p_arch.fit_transform(train_ds)
    #     test_ds = pca_dlv3p_arch.transform(test_ds)
    #     apply_pca_ds(train_ds, test_ds, )
    # elif dnn_architecture == "probunet":
    #     pass
    # else:
    #     raise ValueError(" DNN architecture not found! Choose a valid DNN architecture!")

    # labels
    label_test_ind_ds = np.zeros((ind_valid_data.shape[0], 1))  # 0
    label_test_ood_ds = np.ones((ood_valid_data.shape[0], 1))  # 1
    labels_train_ds = np.vstack((label_test_ind_ds, label_test_ood_ds))
    labels_train_ds = labels_train_ds.astype('int32')
    labels_train_ds = np.squeeze(labels_train_ds)

    label_test_ind_ds = np.zeros((ind_test_data.shape[0], 1))  # 0
    label_test_ood_ds = np.ones((ood_test_data.shape[0], 1))  # 1
    labels_test_ds = np.vstack((label_test_ind_ds, label_test_ood_ds))
    labels_test_ds = labels_test_ds.astype('int32')
    labels_test_ds = np.squeeze(labels_test_ds)

    print("Train Dataset Samples shape: ", train_ds.shape)
    print("Train Dataset Labels shape: ", labels_train_ds.shape)

    print("Test Dataset Samples shape: ", test_ds.shape)
    print("Test Dataset Labels shape: ", labels_test_ds.shape)

    return train_ds, labels_train_ds, test_ds, labels_test_ds
