from unittest import TestCase
from ..ls_ood_detect_cea.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform
from ..ls_ood_detect_cea.metrics import *


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
        self.assertAlmostEqual(results["fpr@95"].values[0], 0.7329999804496765)
        self.assertAlmostEqual(results["aupr"].values[0], 0.7484172582626343)
        self.assertAlmostEqual(results["auroc"].values[0], 0.7622030377388)

    def test_evaluate_lared_larem(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        train_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        ood_ds_name = "test"
        ood_dict = {ood_ds_name: -0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)}
        results = log_evaluate_lared_larem(
            ind_train_h_z=train_ind,
            ind_test_h_z=test_ind,
            ood_h_z_dict=ood_dict
        )
        self.assertAlmostEqual(
            results.loc[f"{ood_ds_name} LaRED"]["auroc"],
            0.9425060153007507,
            delta=TOL
        )
        self.assertAlmostEqual(
            results.loc[f"{ood_ds_name} LaRED"]["aupr"],
            0.9419655799865723,
            delta=TOL
        )
        self.assertAlmostEqual(
            results.loc[f"{ood_ds_name} LaRED"]["fpr@95"],
            0.2759999930858612,
            delta=TOL
        )
        self.assertAlmostEqual(
            results.loc[f"{ood_ds_name} LaREM"]["auroc"],
            0.9485160112380981,
            delta=TOL
        )
        self.assertAlmostEqual(
            results.loc[f"{ood_ds_name} LaREM"]["aupr"],
            0.9480809569358826,
            delta=TOL
        )
        self.assertAlmostEqual(
            results.loc[f"{ood_ds_name} LaREM"]["fpr@95"],
            0.25600001215934753,
            delta=TOL
        )

    def test_select_best_lared_larem(self):
        np.random.seed(SEED)
        test_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        train_ind = 0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)
        ood_ds_name = "test"
        ood_dict = {ood_ds_name: -0.5 + np.random.randn(TEST_SET_SIZE, LATENT_SPACE_DIM)}
        pca_components = (2, 6, 10)
        overall_metrics_df = pd.DataFrame(columns=['auroc', 'fpr@95', 'aupr',
                                                   'fpr', 'tpr', 'roc_thresholds',
                                                   'precision', 'recall', 'pr_thresholds'])
        for n_components in pca_components:
            # Perform PCA dimension reduction
            pca_h_z_ind_train, pca_transformation = apply_pca_ds_split(
                samples=train_ind,
                nro_components=n_components
            )
            pca_h_z_ind_test = apply_pca_transform(test_ind, pca_transformation)
            ood_pca_dict = {}
            ood_pca_dict[ood_ds_name] = apply_pca_transform(ood_dict[ood_ds_name],
                                                            pca_transformation)

            r_df = log_evaluate_lared_larem(
                ind_train_h_z=pca_h_z_ind_train,
                ind_test_h_z=pca_h_z_ind_test,
                ood_h_z_dict=ood_pca_dict,
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores=False,
                log_step=n_components,
                mlflow_logging=False
            )
            # Add results to df
            overall_metrics_df = overall_metrics_df.append(r_df)
        # Check LaRED results
        auroc_lared, aupr_lared, fpr_lared = select_and_log_best_lared_larem(
            overall_metrics_df, pca_components, technique="LaRED", log_mlflow=False
        )
        self.assertAlmostEqual(auroc_lared, 0.8123379945755005)
        self.assertAlmostEqual(aupr_lared, 0.7958850860595703)
        self.assertAlmostEqual(fpr_lared, 0.5989999771118164)

        # Check LaREM results
        auroc_larem, aupr_larem, fpr_larem = select_and_log_best_lared_larem(
            overall_metrics_df, pca_components, technique="LaREM", log_mlflow=True
        )
        self.assertAlmostEqual(auroc_larem, 0.8106609582901001)
        self.assertAlmostEqual(aupr_larem, 0.7947214841842651)
        self.assertAlmostEqual(fpr_larem, 0.6159999966621399)



