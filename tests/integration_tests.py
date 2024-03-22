from unittest import TestCase, main
from ls_ood_detect_cea.uncertainty_estimation import *
from ls_ood_detect_cea.metrics import *
from ls_ood_detect_cea.dimensionality_reduction import apply_pca_ds_split, apply_pca_transform
from tests_architecture import Net
import torchvision
import urllib
import tarfile
import os

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_PROPORTION = 0.1
MCD_N_SAMPLES = 3
LATENT_SPACE_DIM = 20
TOL = 1e-7
N_PCA_COMPONENTS = 4
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
if not os.path.exists("./emnist_data_source/EMNIST/raw"):
    print("getting emnist raw data from confianceai repository ..")
    if not os.path.exists("./emnist_data_source/EMNIST"):
        os.makedirs("./emnist_data_source/EMNIST")

    urllib.request.urlretrieve(
        "https://minio-storage.apps.confianceai-public.irtsysx.fr/ml-models/emnist.tar.gz",
        "./emnist_data_source/EMNIST/emnist.tar.gz",
    )

    file = tarfile.open("./emnist_data_source/EMNIST/emnist.tar.gz")
    file.extractall("./emnist_data_source/EMNIST")
    file.close()
    print("emnist raw data have been downloaded")

emnist_data = torchvision.datasets.EMNIST(
    "./emnist_data_source", split="letters", train=False, download=False, transform=transforms
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
        ood_ds_name = "test_ood"
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

            r_df, ind_larem_score, ood_larem_scores_dict = log_evaluate_lared_larem(
                ind_train_h_z=pca_h_z_ind_train,
                ind_test_h_z=pca_h_z_ind_test,
                ood_h_z_dict=ood_pca_dict,
                experiment_name_extension=f" PCA {n_components}",
                return_density_scores=True,
                log_step=n_components,
                mlflow_logging=False,
            )
            # Add results to df
            overall_metrics_df = overall_metrics_df.append(r_df)
        # Check LaRED results
        auroc_lared, aupr_lared, fpr_lared, best_n_comps_lared = select_and_log_best_lared_larem(
            overall_metrics_df, pca_components, technique="LaRED", log_mlflow=False
        )
        self.assertAlmostEqual(0.8123379945755005, auroc_lared, delta=TOL)
        self.assertAlmostEqual(0.7958850860595703, aupr_lared, delta=TOL)
        self.assertAlmostEqual(0.5989999771118164, fpr_lared, delta=TOL)

        # Check LaREM results
        auroc_larem, aupr_larem, fpr_larem, best_n_comps_larem = select_and_log_best_lared_larem(
            overall_metrics_df, pca_components, technique="LaREM", log_mlflow=True
        )
        self.assertAlmostEqual(0.8106609582901001, auroc_larem, delta=TOL)
        self.assertAlmostEqual(0.7947214841842651, aupr_larem, delta=TOL)
        self.assertAlmostEqual(0.6159999966621399, fpr_larem, delta=TOL)

        roc_curve_test = save_roc_ood_detector(overall_metrics_df, "Test title")
        self.assertEqual(1.0, roc_curve_test.axes[0].dataLim.max[0])
        self.assertEqual(1.0, roc_curve_test.axes[0].dataLim.max[1])
        self.assertEqual(0.0, roc_curve_test.axes[0].dataLim.min[0])
        self.assertEqual(0.0, roc_curve_test.axes[0].dataLim.min[1])
        self.assertAlmostEqual(0.0010000000474974513, roc_curve_test.axes[0].dataLim.minposx)
        self.assertAlmostEqual(0.0010000000474974513, roc_curve_test.axes[0].dataLim.minposy)

        experiment_dict = {
            "InD": ind_larem_score,
            "x_axis": "LaREM score",
            "plot_name": "LaREM test plot",
            "test_ood": ood_larem_scores_dict[ood_ds_name],
        }
        pred_scores_plot_test = get_pred_scores_plots(
            experiment=experiment_dict,
            ood_datasets_list=[ood_ds_name],
            title="Test title",
            ind_dataset_name="Test InD",
        )
        self.assertAlmostEqual(478.9049999999999, pred_scores_plot_test.ax.bbox.max[0])
        self.assertAlmostEqual(484.9999999999999, pred_scores_plot_test.ax.bbox.max[1], delta=TOL)
        self.assertAlmostEqual(70.65277777777779, pred_scores_plot_test.ax.bbox.min[0], delta=TOL)
        self.assertAlmostEqual(58.277777777777764, pred_scores_plot_test.ax.bbox.min[1], delta=TOL)

    def test_extract_entropy_lared_larem(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(apply_dropout)  # enable dropout
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

    def test_larex_inference(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        ind_test_features = np.random.rand(ind_subset_ds_len, LATENT_SPACE_DIM)
        pca_ind_train, pca_transformation = apply_pca_ds_split(
            samples=ind_test_features, nro_components=N_PCA_COMPONENTS
        )
        larem_processor = LaREMPostprocessor()
        larem_processor.setup(pca_ind_train)
        larem_inference = LaRExInference(
            dnn_model=tests_model,
            detector=larem_processor,
            mcd_sampler=MCSamplerModule,
            pca_transform=pca_transformation,
            mcd_samples_nro=MCD_N_SAMPLES,
            layer_type="Conv",
        )
        ood_iterator = iter(ood_test_loader)
        ood_test_image = next(ood_iterator)[0]
        ood_prediction, ood_img_score = larem_inference.get_score(
            ood_test_image, layer_hook=hooked_layer
        )
        self.assertAlmostEqual(-6103.11052918, ood_img_score, delta=TOL)
        self.assertAlmostEqual(
            0.0,
            (
                ood_prediction[0].cpu().numpy()
                - np.array(
                    [
                        -2.275621,
                        -2.007265,
                        -2.4919932,
                        -2.2528067,
                        -2.1876812,
                        -2.345544,
                        -2.314673,
                        -2.4217446,
                        -2.4728994,
                        -2.3519247,
                    ]
                )
            ).sum(),
            delta=1e-6,
        )


if __name__ == "__main__":
    main()
