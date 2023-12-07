from unittest import TestCase, main
from ls_ood_detect_cea.metrics import *


#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_SIZE = 1000
LATENT_SPACE_DIM = 20
TOL = 1e-7
########################################################################


class Test(TestCase):
    def test_hz_detector_results(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE)
        test_ood = -0.5 + np.random.randn(TEST_SET_SIZE)
        test_name = "test"
        results = get_hz_detector_results(test_name, test_ind, test_ood, False)
        self.assertAlmostEqual(0.7329999804496765, results["fpr@95"].values[0])
        self.assertAlmostEqual(0.7484172582626343, results["aupr"].values[0])
        self.assertAlmostEqual(0.7622030377388, results["auroc"].values[0])

    def test_evaluate_lared_larem(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        train_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        ood_ds_name = "test"
        ood_dict = {ood_ds_name: -0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)}
        results = log_evaluate_lared_larem(
            ind_train_h_z=train_ind, ind_test_h_z=test_ind, ood_h_z_dict=ood_dict
        )
        self.assertAlmostEqual(
            0.9425060153007507, results.loc[f"{ood_ds_name} LaRED"]["auroc"], delta=TOL
        )
        self.assertAlmostEqual(
            0.9419655799865723, results.loc[f"{ood_ds_name} LaRED"]["aupr"], delta=TOL
        )
        self.assertAlmostEqual(
            0.2759999930858612, results.loc[f"{ood_ds_name} LaRED"]["fpr@95"], delta=TOL
        )
        self.assertAlmostEqual(
            0.9485160112380981, results.loc[f"{ood_ds_name} LaREM"]["auroc"], delta=TOL
        )
        self.assertAlmostEqual(
            0.9480809569358826, results.loc[f"{ood_ds_name} LaREM"]["aupr"], delta=TOL
        )
        self.assertAlmostEqual(
            0.25600001215934753, results.loc[f"{ood_ds_name} LaREM"]["fpr@95"], delta=TOL
        )


if __name__ == "__main__":
    main()
