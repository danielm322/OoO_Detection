from unittest import TestCase, main
from ls_ood_detect_cea.uncertainty_estimation import *
from tests_architecture import Net
import torch
import torchvision

#########################################################################
# PARAMETERS
# If you change any of the following parameters the tests will not pass!!
SEED = 1
TEST_SET_PROPORTION = 0.1
MCD_N_SAMPLES = 3
LATENT_SPACE_DIM = 20
TOL = 1e-6
LAYER_TYPE = "Conv"
REDUCTION_METHOD = "fullmean"
########################################################################

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset for testing
mnist_data = torchvision.datasets.MNIST(
    "./mnist-data/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
subset_ds_len = int(len(mnist_data) * TEST_SET_PROPORTION)
test_subset = torch.utils.data.random_split(
    mnist_data,
    [subset_ds_len, len(mnist_data) - subset_ds_len],
    torch.Generator().manual_seed(SEED),
)[0]
# DataLoader
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)
# Define toy model for testing
tests_model = Net(latent_space_dimension=LATENT_SPACE_DIM)
tests_model.to(device)
tests_model.eval()


class Test(TestCase):
    ############################################
    # Uncertainty estimation Tests
    ############################################
    def test_hook_module(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model(next(iter(test_loader))[0].to(device))
        self.assertEqual(hooked_layer.output.shape, torch.Size([1, LATENT_SPACE_DIM, 8, 8]))
        self.assertTrue(
            (
                hooked_layer.output[0, 0, 0].cpu()
                - torch.Tensor(
                    [
                        0.0800926983,
                        0.2299261838,
                        0.1115054339,
                        0.4455563724,
                        0.1744501591,
                        0.3627213538,
                        0.4317324460,
                        0.0737401918,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_latent_representation_mcd_samples(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(apply_dropout)  # enable dropout
        mcd_samples = get_latent_representation_mcd_samples(
            tests_model, test_loader, MCD_N_SAMPLES, hooked_layer, "Conv"
        )
        self.assertEqual(
            mcd_samples.shape, torch.Size([MCD_N_SAMPLES * subset_ds_len, LATENT_SPACE_DIM])
        )
        self.assertTrue(
            (
                mcd_samples[0].cpu()
                - torch.Tensor(
                    [
                        0.0032064617,
                        0.1210283488,
                        0.0136697628,
                        0.1224055067,
                        -0.0258768126,
                        0.0873960480,
                        -0.0569284856,
                        -0.0059729815,
                        -0.5415794849,
                        0.0916501209,
                        -0.6159725785,
                        -0.0845529139,
                        -0.6451811790,
                        0.0113038644,
                        -0.4393664598,
                        -0.0969015583,
                        -0.5628786683,
                        0.8988993168,
                        -0.1274375170,
                        0.2650382817,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_mcd_samples_extractor(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.conv2_drop)
        tests_model.apply(apply_dropout)  # enable dropout
        samples_extractor = MCDSamplesExtractor(
            model=tests_model,
            mcd_nro_samples=MCD_N_SAMPLES,
            hooked_layer=hooked_layer,
            layer_type=LAYER_TYPE,
            device=device,
            reduction_method=REDUCTION_METHOD,
            return_raw_predictions=False,
        )
        mcd_samples = samples_extractor.get_ls_mcd_samples(data_loader=test_loader)
        self.assertEqual(
            mcd_samples.shape, torch.Size([MCD_N_SAMPLES * subset_ds_len, LATENT_SPACE_DIM])
        )
        self.assertTrue(
            (
                mcd_samples[0].cpu()
                - torch.Tensor(
                    [
                        0.0032064617,
                        0.1210283488,
                        0.0136697628,
                        0.1224055067,
                        -0.0258768126,
                        0.0873960480,
                        -0.0569284856,
                        -0.0059729815,
                        -0.5415794849,
                        0.0916501209,
                        -0.6159725785,
                        -0.0845529139,
                        -0.6451811790,
                        0.0113038644,
                        -0.4393664598,
                        -0.0969015583,
                        -0.5628786683,
                        0.8988993168,
                        -0.1274375170,
                        0.2650382817,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_single_image_entropy_calculation(self):
        np.random.seed(SEED)
        sample_random_image = np.random.rand(MCD_N_SAMPLES, LATENT_SPACE_DIM)
        single_image_entropy = single_image_entropy_calculation(
            sample_random_image, MCD_N_SAMPLES - 1
        )
        self.assertEqual(single_image_entropy.shape, (LATENT_SPACE_DIM,))
        self.assertTrue(
            np.allclose(
                single_image_entropy,
                np.array(
                    [
                        0.50127026,
                        -0.24113694,
                        -0.00449107,
                        0.3995355,
                        0.91656596,
                        0.77765932,
                        0.9553056,
                        -0.05127198,
                        -0.50808706,
                        0.70149445,
                        0.20306963,
                        -0.14639144,
                        0.90684774,
                        0.5116368,
                        0.66483277,
                        0.52607873,
                        -0.29929704,
                        0.64812784,
                        0.55261872,
                        0.56711444,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_get_dl_h_z(self):
        torch.manual_seed(SEED)
        test_latent_rep = torch.rand(MCD_N_SAMPLES * subset_ds_len, LATENT_SPACE_DIM)
        test_entropy = get_dl_h_z(test_latent_rep, MCD_N_SAMPLES, parallel_run=True)[1]
        self.assertEqual(test_entropy.shape, (subset_ds_len, LATENT_SPACE_DIM))
        self.assertTrue(
            np.allclose(
                test_entropy[0],
                np.array(
                    [
                        0.66213845,
                        -1.24553263,
                        0.43274342,
                        0.16964551,
                        -0.11109334,
                        -0.15076459,
                        0.50606536,
                        -0.938645,
                        0.4492352,
                        0.1962833,
                        0.7151791,
                        0.70086446,
                        -0.89067232,
                        0.01402643,
                        0.96637271,
                        0.6283722,
                        0.34451879,
                        -0.00393629,
                        -0.48039907,
                        0.2459123,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_mcd_pred_uncertainty_score(self):
        torch.manual_seed(SEED)
        samples, pred_h, mi = get_mcd_pred_uncertainty_score(
            dnn_model=tests_model, input_dataloader=test_loader, mcd_nro_samples=MCD_N_SAMPLES
        )
        self.assertEqual(
            samples.shape, torch.Size([subset_ds_len, MCD_N_SAMPLES, len(mnist_data.classes)])
        )
        self.assertTrue(
            (
                samples[0, 0].cpu()
                - torch.Tensor(
                    [
                        0.1046028882,
                        0.1448072642,
                        0.0831882283,
                        0.0958161354,
                        0.1245612875,
                        0.0838165581,
                        0.0954214483,
                        0.0930665582,
                        0.0805836692,
                        0.0941359326,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(pred_h.shape, torch.Size([subset_ds_len]))
        self.assertTrue(
            (
                pred_h[:LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        2.2853562832,
                        2.2832891941,
                        2.2903933525,
                        2.2847099304,
                        2.2912006378,
                        2.2932782173,
                        2.2862076759,
                        2.2852103710,
                        2.2850139141,
                        2.2867050171,
                        2.2891502380,
                        2.2882592678,
                        2.2852959633,
                        2.2851634026,
                        2.2892885208,
                        2.2856822014,
                        2.2813849449,
                        2.2889823914,
                        2.2855486870,
                        2.2835049629,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(mi.shape, torch.Size([subset_ds_len]))
        self.assertTrue(
            (
                mi[:LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        0.0000000000e00,
                        -2.3841857910e-07,
                        0.0000000000e00,
                        2.3841857910e-07,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        2.3841857910e-07,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                        0.0000000000e00,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_get_pred_uncertainty_score(self):
        torch.manual_seed(SEED)
        input_samples = torch.rand(subset_ds_len * MCD_N_SAMPLES, len(mnist_data.classes))
        pred_h, mi = get_predictive_uncertainty_score(input_samples, MCD_N_SAMPLES)
        self.assertEqual(pred_h.shape, torch.Size([subset_ds_len]))
        self.assertTrue(
            (
                pred_h[:LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        2.2954239845,
                        2.2936100960,
                        2.2913274765,
                        2.2875323296,
                        2.2929368019,
                        2.2918679714,
                        2.2960093021,
                        2.2968566418,
                        2.2959966660,
                        2.2814788818,
                        2.2842481136,
                        2.2999031544,
                        2.2889721394,
                        2.2876074314,
                        2.2898373604,
                        2.2722413540,
                        2.2952365875,
                        2.2890031338,
                        2.2956945896,
                        2.2963838577,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(mi.shape, torch.Size([subset_ds_len]))
        self.assertTrue(
            (
                mi[:LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        0.0174179077,
                        0.0362265110,
                        0.0232129097,
                        0.0191953182,
                        0.0291962624,
                        0.0250487328,
                        0.0286319256,
                        0.0462374687,
                        0.0337688923,
                        0.0085363388,
                        0.0118212700,
                        0.0289292336,
                        0.0174930096,
                        0.0185499191,
                        0.0275359154,
                        0.0115809441,
                        0.0252931118,
                        0.0210585594,
                        0.0264883041,
                        0.0213701725,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_get_msp_score(self):
        torch.manual_seed(SEED)
        msp_score = get_msp_score(tests_model, test_loader)
        self.assertEqual(msp_score.shape, (subset_ds_len,))
        self.assertTrue(
            np.allclose(
                msp_score[:LATENT_SPACE_DIM],
                np.array(
                    [
                        0.14480726,
                        0.13710754,
                        0.13400422,
                        0.14451809,
                        0.13787547,
                        0.13306601,
                        0.14861068,
                        0.14621589,
                        0.14574473,
                        0.13511162,
                        0.13810264,
                        0.14093828,
                        0.1471938,
                        0.144996,
                        0.14435232,
                        0.14504616,
                        0.14830166,
                        0.13930918,
                        0.14155908,
                        0.14461976,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_get_energy_score(self):
        torch.manual_seed(SEED)
        energy_score = get_energy_score(tests_model, test_loader)
        self.assertEqual(energy_score.shape, (subset_ds_len,))
        self.assertTrue(
            np.allclose(
                energy_score[:LATENT_SPACE_DIM],
                np.array(
                    [
                        -1.1920929e-07,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        2.3841858e-07,
                        0.0000000e00,
                        0.0000000e00,
                        1.1920929e-07,
                        0.0000000e00,
                        -1.1920929e-07,
                        1.1920929e-07,
                        1.1920929e-07,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                        0.0000000e00,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_MDSPostprocessor(self):
        torch.manual_seed(SEED)
        hooked_layer = Hook(tests_model.fc1)
        mahalanobis_distance_estimator = MDSPostprocessor(num_classes=len(mnist_data.classes))
        mahalanobis_distance_estimator.setup(tests_model, test_loader, hooked_layer)
        self.assertEqual(mahalanobis_distance_estimator.precision.shape, torch.Size([50, 50]))
        self.assertTrue(
            (
                mahalanobis_distance_estimator.precision[0, :LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        165.2596282959,
                        -12.5079212189,
                        8.9375352859,
                        -3.4928135872,
                        6.0702023506,
                        -17.6381549835,
                        -1.1519905329,
                        14.5131769180,
                        -36.5683860779,
                        30.9377803802,
                        -5.0159239769,
                        30.1570549011,
                        -9.7787103653,
                        11.7463140488,
                        -5.8539495468,
                        -2.8339817524,
                        -1.1148993969,
                        -21.3802356720,
                        -43.9337997437,
                        -15.5770769119,
                    ]
                )
            ).sum()
            < TOL
        )
        self.assertEqual(mahalanobis_distance_estimator.class_mean.shape, torch.Size([10, 50]))
        self.assertTrue(
            (
                mahalanobis_distance_estimator.class_mean[0, :LATENT_SPACE_DIM].cpu()
                - torch.Tensor(
                    [
                        0.5837824345,
                        0.0068388470,
                        0.2311379015,
                        0.3187698424,
                        0.2801628113,
                        0.0079951715,
                        0.0159174353,
                        -0.5741344094,
                        0.1619823128,
                        -0.6143783927,
                        0.2275377214,
                        -0.2329114527,
                        -0.0491828360,
                        0.1668033451,
                        0.0840948597,
                        0.0450854897,
                        0.0546373986,
                        0.0204664711,
                        -0.3509032130,
                        0.0088814823,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_kNNPostprocessor(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        hooked_layer = Hook(tests_model.fc1)
        knn_processor = KNNPostprocessor(K=50)
        knn_processor.setup(tests_model, test_loader, hooked_layer)
        self.assertEqual(knn_processor.activation_log.shape, (subset_ds_len, 50))
        self.assertTrue(
            (
                knn_processor.activation_log[0, :LATENT_SPACE_DIM]
                - np.array(
                    [
                        0.35141832,
                        0.08980233,
                        0.03794549,
                        0.11652409,
                        0.12736143,
                        0.06724516,
                        0.02734577,
                        -0.18319036,
                        0.01246245,
                        -0.39833596,
                        0.14618297,
                        -0.0928729,
                        0.03393501,
                        0.12940583,
                        0.15052024,
                        0.13174318,
                        0.08799291,
                        0.14757843,
                        -0.08823613,
                        0.10692727,
                    ]
                )
            ).sum()
            < TOL
        )
        post_processed = knn_processor.postprocess(tests_model, test_loader, hooked_layer)[1]
        self.assertEqual(post_processed.shape, (subset_ds_len,))
        self.assertTrue(
            (
                post_processed[:LATENT_SPACE_DIM]
                - np.array(
                    [
                        -0.22327094,
                        -0.15341546,
                        -0.24850407,
                        -0.21372338,
                        -0.15940219,
                        -0.18464218,
                        -0.14856,
                        -0.18642662,
                        -0.17184329,
                        -0.1617361,
                        -0.15050437,
                        -0.15412888,
                        -0.23904872,
                        -0.20866983,
                        -0.2049842,
                        -0.19164929,
                        -0.19247529,
                        -0.14494337,
                        -0.3010658,
                        -0.27074164,
                    ]
                )
            ).sum()
            < TOL
        )

    def test_larem_postprocessor(self):
        np.random.seed(SEED)
        test_features = np.random.rand(subset_ds_len, LATENT_SPACE_DIM)
        larem_processor = LaREMPostprocessor()
        larem_processor.setup(test_features)
        self.assertEqual(larem_processor.precision.shape, (LATENT_SPACE_DIM, LATENT_SPACE_DIM))
        self.assertTrue(
            np.allclose(
                larem_processor.precision[0],
                np.array(
                    [
                        12.61048445,
                        -0.75305318,
                        -0.50165124,
                        -0.11791143,
                        -0.11522137,
                        0.01881437,
                        -0.67406623,
                        0.4608969,
                        -0.46416246,
                        -0.02338144,
                        0.1637525,
                        -0.27044667,
                        -0.56074832,
                        0.1450181,
                        0.25837548,
                        0.84070141,
                        0.6010112,
                        -0.15461955,
                        -0.34001486,
                        0.24692961,
                    ]
                ),
                atol=TOL,
            )
        )
        postprocessed = larem_processor.postprocess(test_features)
        self.assertEqual(postprocessed.shape, (subset_ds_len,))
        self.assertTrue(
            np.allclose(
                postprocessed[:LATENT_SPACE_DIM],
                np.array(
                    [
                        -18.57565347,
                        -25.08741653,
                        -19.42807739,
                        -20.54087356,
                        -20.43976452,
                        -26.15764235,
                        -25.1002785,
                        -26.32450315,
                        -16.85869173,
                        -26.99430034,
                        -24.48394177,
                        -16.25644187,
                        -20.19520139,
                        -12.63947527,
                        -17.54776882,
                        -17.30209618,
                        -15.89407861,
                        -20.50299392,
                        -24.93608634,
                        -16.28581699,
                    ]
                ),
                atol=TOL,
            )
        )

    def test_lared_postprocessor(self):
        np.random.seed(SEED)
        test_features = np.random.rand(subset_ds_len, LATENT_SPACE_DIM)
        lared_processor = LaREDPostprocessor()
        lared_processor.setup(test_features)
        postprocessed = lared_processor.postprocess(test_features)
        self.assertEqual(postprocessed.shape, (subset_ds_len,))
        self.assertTrue(
            np.allclose(
                postprocessed[:LATENT_SPACE_DIM],
                np.array(
                    [
                        -19.91285992,
                        -20.18685465,
                        -19.98303992,
                        -19.937434,
                        -19.92117358,
                        -20.22682757,
                        -20.14255507,
                        -20.2227513,
                        -19.82865285,
                        -20.13994971,
                        -20.19335801,
                        -19.86726602,
                        -19.97992749,
                        -19.664516,
                        -19.88776353,
                        -19.88357531,
                        -19.76506903,
                        -20.0045806,
                        -20.1114499,
                        -19.82522835,
                    ]
                ),
                atol=TOL,
            )
        )


if __name__ == "__main__":
    main()
