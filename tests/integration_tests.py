from unittest import TestCase, main
from ls_ood_detect_cea.uncertainty_estimation import *
from ls_ood_detect_cea.metrics import *
from ls_ood_detect_cea.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform
from tests_architecture import Net
import torchvision


#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_PROPORTION = 0.5
MCD_N_SAMPLES = 3
LATENT_SPACE_DIM = 20
TOL = 1e-7
########################################################################

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define datasets for testing
transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)
mnist_data = torchvision.datasets.MNIST(
    "./mnist-data/",
    train=False,
    download=True,
    transform=transforms,
)
emnist_data = torchvision.datasets.EMNIST(
    "./emnist-data", split="letters", train=False, download=True, transform=transforms
)
# Subset InD dataset
ind_subset_ds_len = int(len(mnist_data) * TEST_SET_PROPORTION)
ind_test_subset = torch.utils.data.random_split(
    mnist_data,
    [ind_subset_ds_len, len(mnist_data) - ind_subset_ds_len],
    torch.Generator().manual_seed(SEED),
)[0]
# Subset OoD dataset
ood_subset_ds_len = int(len(emnist_data) * TEST_SET_PROPORTION * 0.5)
ood_test_subset = torch.utils.data.random_split(
    emnist_data,
    [ood_subset_ds_len, len(emnist_data) - ood_subset_ds_len],
    torch.Generator().manual_seed(SEED),
)[0]

# DataLoaders
ind_test_loader = torch.utils.data.DataLoader(ind_test_subset, batch_size=1, shuffle=True)
ood_test_loader = torch.utils.data.DataLoader(ood_test_subset, batch_size=1, shuffle=True)
# Define toy model for testing
tests_model = Net(latent_space_dimension=LATENT_SPACE_DIM)
tests_model.to(device)
tests_model.eval()


class Test(TestCase):
    def test_select_best_lared_larem(self):
        np.random.seed(SEED)
        # Here we start from a supposed already calculated entropy
        test_ind = 0.5 + np.random.randn(ind_subset_ds_len, LATENT_SPACE_DIM)
        train_ind = 0.5 + np.random.randn(ind_subset_ds_len, LATENT_SPACE_DIM)
        ood_ds_name = "test"
        ood_dict = {ood_ds_name: -0.5 + np.random.randn(ind_subset_ds_len, LATENT_SPACE_DIM)}
        pca_components = (2, 6, 10)
        overall_metrics_df = pd.DataFrame(
            columns=[
                "auroc",
                "fpr@95",
                "aupr",
                "fpr",
                "tpr",
                "roc_thresholds",
                "precision",
                "recall",
                "pr_thresholds",
            ]
        )
        for n_components in pca_components:
            # Perform PCA dimension reduction
            pca_h_z_ind_train, pca_transformation = apply_pca_ds_split(
                samples=train_ind, nro_components=n_components
            )
            pca_h_z_ind_test = apply_pca_transform(test_ind, pca_transformation)
            ood_pca_dict = {}
            ood_pca_dict[ood_ds_name] = apply_pca_transform(
                ood_dict[ood_ds_name], pca_transformation
            )

            r_df = log_evaluate_lared_larem(
                ind_train_h_z=pca_h_z_ind_train,
                ind_test_h_z=pca_h_z_ind_test,
                ood_h_z_dict=ood_pca_dict,
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores=False,
                log_step=n_components,
                mlflow_logging=False,
            )
            # Add results to df
            overall_metrics_df = overall_metrics_df.append(r_df)
        # Check LaRED results
        auroc_lared, aupr_lared, fpr_lared = select_and_log_best_lared_larem(
            overall_metrics_df, pca_components, technique="LaRED", log_mlflow=False
        )
        self.assertAlmostEqual(0.9400724172592163, auroc_lared)
        self.assertAlmostEqual(0.9367987513542175, aupr_lared)
        self.assertAlmostEqual(0.25679999589920044, fpr_lared)

        # Check LaREM results
        auroc_larem, aupr_larem, fpr_larem = select_and_log_best_lared_larem(
            overall_metrics_df, pca_components, technique="LaREM", log_mlflow=True
        )
        self.assertAlmostEqual(0.9411856532096863, auroc_larem)
        self.assertAlmostEqual(0.938223659992218, aupr_larem)
        self.assertAlmostEqual(0.2556000053882599, fpr_larem)

    def test_extract_entropy_lared_larem(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(deeplabv3p_apply_dropout)  # enable dropout
        ind_mcd_latent_samples = get_latent_representation_mcd_samples(
            tests_model, ind_test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        ood_mcd_latent_samples = get_latent_representation_mcd_samples(
            tests_model, ood_test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        ind_test_entropy = get_dl_h_z(ind_mcd_latent_samples, MCD_N_SAMPLES, parallel_run=True)[1]
        ood_test_entropy = get_dl_h_z(ood_mcd_latent_samples, MCD_N_SAMPLES, parallel_run=True)[1]
        self.assertEqual((ind_subset_ds_len, LATENT_SPACE_DIM), ind_test_entropy.shape)
        self.assertEqual((ood_subset_ds_len, LATENT_SPACE_DIM), ood_test_entropy.shape)
        self.assertAlmostEqual(
            0.0,
            (
                ind_test_entropy[0]
                - np.array(
                    [
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                    ]
                )
            ).sum(),
            delta=TOL,
        )
        self.assertAlmostEqual(
            0.0,
            (
                ood_test_entropy[0]
                - np.array(
                    [
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                        -10.31977828,
                    ]
                )
            ).sum(),
            delta=TOL,
        )


if __name__ == "__main__":
    main()
