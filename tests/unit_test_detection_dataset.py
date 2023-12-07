from unittest import TestCase, main
from ls_ood_detect_cea.ood_detection_dataset import *

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
    def test_build_ood_detection_ds(self):
        np.random.seed(SEED)
        ind_valid_data = 0.5 + np.random.randn(TEST_SET_SIZE + 1, LATENT_SPACE_DIM)
        ood_valid_data = 0.5 + np.random.randn(TEST_SET_SIZE - 1, LATENT_SPACE_DIM)
        ind_test_data = -0.5 + np.random.randn(TEST_SET_SIZE + 2, LATENT_SPACE_DIM)
        ood_test_data = -0.5 + np.random.randn(TEST_SET_SIZE - 2, LATENT_SPACE_DIM)
        train_ds, labels_train_ds, test_ds, labels_test_ds, pca_dim_red = build_ood_detection_ds(
            ind_valid_data, ood_valid_data, ind_test_data, ood_test_data, pca_nro_comp=N_PCA_COMPS
        )
        self.assertEqual(
            (ind_valid_data.shape[0] + ood_valid_data.shape[0], N_PCA_COMPS), train_ds.shape
        )
        self.assertEqual(
            (ind_test_data.shape[0] + ood_test_data.shape[0], N_PCA_COMPS), test_ds.shape
        )
        self.assertEqual((N_PCA_COMPS, LATENT_SPACE_DIM), pca_dim_red.components_.shape)
        self.assertAlmostEqual(
            0.0,
            (
                pca_dim_red.components_[0]
                - np.array(
                    [
                        0.53244494,
                        -0.29243398,
                        -0.13776612,
                        0.15066367,
                        0.21902923,
                        -0.00132002,
                        0.07600957,
                        0.0706225,
                        -0.1647277,
                        0.34892125,
                        0.13560772,
                        0.01844636,
                        -0.12813215,
                        -0.02378675,
                        -0.03386188,
                        -0.23064498,
                        0.49739861,
                        -0.0573299,
                        0.16606249,
                        -0.11557661,
                    ]
                )
            ).sum(),
            delta=TOL,
        )

    def test_build_ood_detection_train_split(self):
        np.random.seed(SEED)
        ind_data = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        ood_data = -0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        samples, labels, pca_dim_red = build_ood_detection_train_split(
            ind_data, ood_data, pca_nro_comp=N_PCA_COMPS
        )
        self.assertAlmostEqual(
            0.0,
            (
                pca_dim_red.components_[0]
                - np.array(
                    [
                        -0.21500888,
                        -0.2095538,
                        -0.21595175,
                        -0.23841356,
                        -0.24099829,
                        -0.20559374,
                        -0.23243709,
                        -0.23290823,
                        -0.21720678,
                        -0.21808906,
                        -0.22605419,
                        -0.22942037,
                        -0.21927067,
                        -0.22146763,
                        -0.20630824,
                        -0.20583993,
                        -0.2363615,
                        -0.24386913,
                        -0.22859419,
                        -0.22277621,
                    ]
                )
            ).sum(),
            delta=TOL,
        )


if __name__ == "__main__":
    main()
