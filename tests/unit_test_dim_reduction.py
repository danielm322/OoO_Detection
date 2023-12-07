from unittest import TestCase, main
from ..ls_ood_detect_cea.dimensionality_reduction import *

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_SIZE = 1000
LATENT_SPACE_DIM = 20
TOL = 1e-7
N_PCA_COMPS = 10
########################################################################


class Test(TestCase):
    def test_pca_ds_split(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        transformed, pca_estimator = apply_pca_ds_split(test_ind, N_PCA_COMPS)
        self.assertAlmostEqual(
            (
                transformed[0]
                - np.array(
                    [
                        -2.1572636,
                        -0.02918568,
                        1.06571381,
                        1.0444882,
                        -0.10929565,
                        0.67405348,
                        -1.73276094,
                        -2.06602592,
                        -0.11980209,
                        -1.45960798,
                    ]
                )
            ).sum(),
            0.0,
            delta=TOL,
        )
        self.assertAlmostEqual(
            (
                pca_estimator.components_[0]
                - np.array(
                    [
                        -0.37350362,
                        0.06215473,
                        0.14514634,
                        -0.00179509,
                        -0.23461121,
                        0.01948075,
                        -0.14813394,
                        0.17336065,
                        0.14877849,
                        -0.38446628,
                        -0.3087431,
                        0.1398294,
                        0.00777927,
                        0.12941305,
                        -0.14334455,
                        0.1173632,
                        -0.53262784,
                        0.31606103,
                        0.00491676,
                        0.0926095,
                    ]
                )
            ).sum(),
            0.0,
            delta=TOL,
        )

    def test_apply_pca_transform(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        test_ood = -0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        transformed, pca_estimator = apply_pca_ds_split(test_ind, N_PCA_COMPS)
        ood_transformed = apply_pca_transform(test_ood, pca_estimator)
        self.assertAlmostEqual(
            (
                ood_transformed[0]
                - np.array(
                    [
                        1.99518442,
                        -0.39676575,
                        -1.03689749,
                        0.66995493,
                        -0.54343589,
                        0.63696048,
                        0.64696679,
                        -2.20432657,
                        -0.08940193,
                        0.39293847,
                    ]
                )
            ).sum(),
            0.0,
            delta=TOL,
        )


if __name__ == "__main__":
    main()
