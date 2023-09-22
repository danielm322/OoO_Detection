from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import avg_pool2d
from dropblock import DropBlock2D
import pytorch_lightning as pl
from entropy_estimators import continuous
from icecream import ic
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from copy import deepcopy
import faiss

# from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map


def get_ls_samples_priornet(prob_module: pl.LightningModule, dataloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(device)
    with torch.no_grad():
        dl_mean_samples = []
        dl_var_samples = []
        dl_z_samples = []
        dl_labels = []

        for i, (b_sample_data, b_sample_labels) in enumerate(dataloader):
            b_sample_data = b_sample_data.to(device)
            b_sample_labels = b_sample_labels.to(device)
            prob_module.prob_unet_model.forward(b_sample_data, b_sample_labels, training=False)
            mean_i = prob_module.prob_unet_model.prior_latent_space.mean
            var_i = prob_module.prob_unet_model.prior_latent_space.variance
            dl_mean_samples.append(mean_i)
            dl_var_samples.append(var_i)
            prob_module.prob_unet_model.sample(testing=True)
            z_i = prob_module.prob_unet_model.z_prior_sample
            dl_z_samples.append(z_i)
            # append gt labels
            dl_labels.append(b_sample_labels)

        dl_mean_samples_t = torch.cat(dl_mean_samples, dim=0)
        dl_var_samples_t = torch.cat(dl_var_samples, dim=0)
        dl_z_samples_t = torch.cat(dl_z_samples, dim=0)
        ic(dl_mean_samples_t.shape)
        ic(dl_var_samples_t.shape)
        ic(dl_z_samples_t.shape)

    return dl_z_samples_t, dl_mean_samples_t, dl_var_samples_t


def get_ls_mc_samples_priornet(prob_module, dataloader, mc_samples_nro=32):
    """
    Get Monte Carlo Dropout (MCD) Samples from Probabilistic U-Net - Prior Net
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(device)
    with torch.no_grad():
        dl_mean_samples = []
        dl_variance_samples = []
        dl_z_samples = []
        dl_labels = []

        for b_i, (b_sample_data, b_sample_labels) in enumerate(dataloader):
            b_sample_data = b_sample_data.to(device)
            b_sample_labels = b_sample_labels.to(device)
            mean_mcd_samples = []
            var_mcd_samples = []
            z_mcd_samples = []
            # Get MCD Samples for sample in the batch
            for s_i in range(mc_samples_nro):
                prob_module.prob_unet_model.forward(b_sample_data, b_sample_labels, training=False)
                mean_i = prob_module.prob_unet_model.prior_latent_space.mean
                var_i = prob_module.prob_unet_model.prior_latent_space.variance
                mean_mcd_samples.append(mean_i)
                var_mcd_samples.append(var_i)
                prob_module.prob_unet_model.sample(testing=True)
                z_i = prob_module.prob_unet_model.z_prior_sample
                z_mcd_samples.append(z_i)

            mean_mcd_samples_t = torch.cat(mean_mcd_samples, dim=0)
            var_mcd_samples_t = torch.cat(var_mcd_samples, dim=0)
            z_mcd_samples_t = torch.cat(z_mcd_samples, dim=0)

            dl_mean_samples.append(mean_mcd_samples_t)
            dl_variance_samples.append(var_mcd_samples_t)
            dl_z_samples.append(z_mcd_samples_t)
            # append gt labels
            dl_labels.append(b_sample_labels)

        dl_mean_samples_t = torch.cat(dl_mean_samples, dim=0)
        dl_variance_samples_t = torch.cat(dl_variance_samples, dim=0)
        dl_z_samples_t = torch.cat(dl_z_samples, dim=0)
        dl_labels_t = torch.cat(dl_labels, dim=0)

    return dl_z_samples_t, dl_mean_samples_t, dl_variance_samples_t, dl_labels_t


class Hook:
    """
    Hook class that returns the input and output of a layer during forward/backward pass
    """

    def __init__(self, module: torch.nn.Module, backward: bool = False):
        """
        Hook Class constructor
        :param module: Layer block from Neural Network Module
        :type module: torch.nn.Module
        :param backward: backward-poss hook
        :type backward: bool
        """
        self.input = None
        self.output = None
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def deeplabv3p_get_ls_mcd_samples(
    model_module: pl.LightningModule,
    dataloader: DataLoader,
    mcd_nro_samples: int,
    hook_dropout_layer: Hook,
) -> Tensor:
    """
    Get Monte-Carlo samples form Deeplabv3+ DNN Dropout Layer

    :param model_module: Deeplabv3+ Neural Network Lightning Module
    :type model_module: pl.LightningModule
    :param dataloader: Input samples (torch) Dataloader
    :type dataloader: DataLoader
    :param mcd_nro_samples: Number of Monte-Carlo Samples
    :type mcd_nro_samples: int
    :param hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
    :type hook_dropout_layer: Hook
    :return: Monte-Carlo Dropout samples for the input dataloader
    :rtype: Tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        dl_imgs_latent_mcd_samples = []
        for i, (image, label) in enumerate(dataloader):
            image = image.to(device)
            img_mcd_samples = []
            for s in range(mcd_nro_samples):
                pred_img = model_module.deeplab_v3plus_model(image)
                # pred = torch.argmax(pred_img, dim=1)
                latent_mcd_sample = hook_dropout_layer.output
                # Get image HxW mean:
                latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                # Remove useless dimensions:
                latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)

                img_mcd_samples.append(latent_mcd_sample)

            img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
            dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)

        dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    return dl_imgs_latent_mcd_samples_t


def get_latent_representation_mcd_samples(
    dnn_model: torch.nn.Module,
    dataloader: DataLoader,
    mcd_nro_samples: int,
    layer_hook: Hook,
    layer_type: str,
) -> Tensor:
    """
    Get latent representations Monte-Carlo samples froom DNN using a layer hook

    :param model_module: Neural Network Lightning Module
    :type model_module: pl.LightningModule
    :param dataloader: Input samples (torch) Dataloader
    :type dataloader: DataLoader
    :param mcd_nro_samples: Number of Monte-Carlo Samples
    :type mcd_nro_samples: int
    :param layer_hook: DNN layer hook
    :type layer_hook: Hook
    :param layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or Conv (Convolutional)
    :type: str
    :return: Input dataloader latent representations MC samples tensor
    :rtype: Tensor
    """
    assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as pbar:
            dl_imgs_latent_mcd_samples = []
            for i, (image, label) in enumerate(dataloader):
                image = image.to(device)
                img_mcd_samples = []
                for s in range(mcd_nro_samples):
                    pred_img = dnn_model(image)
                    latent_mcd_sample = layer_hook.output

                    if layer_type == "Conv":
                        # Get image HxW mean:
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                        # Remove useless dimensions:
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                    else:
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes dimension
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=1)

                    img_mcd_samples.append(latent_mcd_sample)

                if layer_type == "Conv":
                    img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                else:
                    img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
                dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    return dl_imgs_latent_mcd_samples_t


def probunet_get_ls_mcd_samples(
    model_module: pl.LightningModule,
    dataloader: DataLoader,
    mcd_nro_samples: int,
    hook_dropout_layer: Hook,
) -> Tensor:
    """
    Get Monte-Carlo samples form ProbUNet DNN Dropout Layer

    :param model_module: ProbUNet Neural Network Lightning Module
    :type model_module: pl.LightningModule
    :param dataloader: Input samples (torch) Dataloader
    :type dataloader: DataLoader
    :param mcd_nro_samples: Number of Monte-Carlo Samples
    :type mcd_nro_samples: int
    :param hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
    :type hook_dropout_layer: Hook
    :return: Monte-Carlo Dropout samples for the input dataloader
    :rtype: Tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        dl_imgs_latent_mcd_samples = []
        for i, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)
            img_mcd_samples = []
            for s in range(mcd_nro_samples):
                model_module.prob_unet_model.forward(image, label, training=False)
                # pred = torch.argmax(pred_img, dim=1)
                latent_mcd_sample = hook_dropout_layer.output
                # Get image HxW mean:
                latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                # Remove useless dimensions:
                latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)

                img_mcd_samples.append(latent_mcd_sample)

            img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
            dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)

        dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    return dl_imgs_latent_mcd_samples_t


def single_image_entropy_calculation(sample: np.array, neighbors: int):
    """
    Function used to calculate the entropy values of a single image. Used to calculate entropy in parallel

    """
    h_z_batch = []
    for z_val_i in range(sample.shape[1]):
        # h_z_i = continuous.get_h(input_mcd_samples[:, z_val_i], k=5)  # old
        h_z_i = continuous.get_h(sample[:, z_val_i], k=neighbors, norm="max", min_dist=1e-5)
        h_z_batch.append(h_z_i)
    h_z_batch_np = np.asarray(h_z_batch)
    return h_z_batch_np


def get_dl_h_z(
    dl_z_samples: Tensor, mcd_samples_nro: int = 32, parallel_run: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataloader Entropy $h(.)$ for Z, from Monte Carlo Dropout (MCD) samples

    :param dl_z_samples: Dataloader Z Samples
    :type dl_z_samples:  Tensor
    :param mcd_samples_nro: Number of monte carlo dropout samples
    :type mcd_samples_nro: int
    :return: Latent vector multivariate normal entropy $h(Z)$, Latent vector value entropy $h(z_i)$
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Get dataloader mvn h(z), from mcd_samples
    z_samples_ls = [i for i in dl_z_samples.split(mcd_samples_nro)]
    # ic(z_samples_ls[0].shape)
    z_samples_np_ls = [t.cpu().numpy() for t in z_samples_ls]
    # ic(z_samples_np_ls[0].shape)
    # dl_h_mvn_z_samples_ls = [continuous.get_h_mvn(s) for s in z_samples_np_ls]
    # Choose correctly the number of neighbors for the entropy calculations:
    # It has to be smaller than the mcd_samples_nro by at least 1
    k_neighbors = 5 if mcd_samples_nro > 5 else mcd_samples_nro - 1
    dl_h_mvn_z_samples_ls = [continuous.get_h(s, k=k_neighbors, norm="max", min_dist=1e-5) for s in z_samples_np_ls]
    dl_h_mvn_z_samples_np = np.array(dl_h_mvn_z_samples_ls)
    dl_h_mvn_z_samples_np = np.expand_dims(dl_h_mvn_z_samples_np, axis=1)
    # ic(dl_h_mvn_z_samples_np.shape)
    # Get dataloader entropy $h(z_i)$ for each value of Z, from mcd_samples
    if not parallel_run:
        dl_h_z_samples = []
        for input_mcd_samples in tqdm(z_samples_np_ls, desc="Calculating entropy"):
            h_z_batch = []
            for z_val_i in range(input_mcd_samples.shape[1]):
                # h_z_i = continuous.get_h(input_mcd_samples[:, z_val_i], k=5)  # old
                h_z_i = continuous.get_h(input_mcd_samples[:, z_val_i], k=k_neighbors, norm="max", min_dist=1e-5)
                h_z_batch.append(h_z_i)
            h_z_batch_np = np.asarray(h_z_batch)
            dl_h_z_samples.append(h_z_batch_np)
    else:
        dl_h_z_samples = process_map(
            single_image_entropy_calculation, z_samples_np_ls, [k_neighbors] * len(z_samples_np_ls), chunksize=1
        )
        # dl_h_z_samples = Parallel(n_jobs=4)(delayed(single_image_entropy_calculation)(i, k_neighbors) for i in z_samples_np_ls)
    dl_h_z_samples_np = np.asarray(dl_h_z_samples)
    # ic(dl_h_z_samples_np.shape)
    return dl_h_mvn_z_samples_np, dl_h_z_samples_np


def probunet_apply_dropout(m):
    if type(m) == torch.nn.Dropout or type(m) == DropBlock2D:
        m.train()


def deeplabv3p_apply_dropout(m):
    if type(m) == torch.nn.Dropout or type(m) == DropBlock2D:
        m.train()


class MCDSamplesExtractor:
    def __init__(
        self,
        model,
        mcd_nro_samples: int,
        hook_dropout_layer: Hook,
        layer_type: str,
        device: str,
        architecture: str,
        location: int,
        reduction_method: str,
        input_size: int,
        original_resnet_architecture: bool = False,
        return_raw_predictions: bool = False,
    ):
        """
        Get Monte-Carlo samples from any torch model Dropout or Dropblock Layer
        :param model: Torch model
        :type model: torch.nn.Module
        :param mcd_nro_samples: Number of Monte-Carlo Samples
        :type mcd_nro_samples: int
        :param hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
        :type hook_dropout_layer: Hook
        :param layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or Conv (Convolutional)
        :type: str
        :param architecture: The model architecture: either small or resnet
        :param location: Location of the hook. This can be useful to select different latent sample catching layers
        :param reduction_method: Whether to use fullmean, mean, or
            avgpool to reduce dimensionality of hooked representation
        :type reduction_method: str
        :param return_raw_predictions: Return or not network outputs
        :return: Monte-Carlo Dropout samples for the input dataloader
        :rtype: Tensor
        """

        assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
        assert architecture in ("small", "resnet"), "Only 'small' or 'resnet' are supported"
        if architecture == "resnet":
            assert input_size in (32, 64, 128)
        if architecture == "resnet" and location in (1, 2):
            assert reduction_method in (
                "mean",
                "fullmean",
                "avgpool",
            ), "Only mean, fullmean and avg pool reduction method supported for resnet"
        self.model = model
        self.mcd_nro_samples = mcd_nro_samples
        self.hook_dropout_layer = hook_dropout_layer
        self.layer_type = layer_type
        self.device = device
        self.architecture = architecture
        self.location = location
        self.reduction_method = reduction_method
        self.input_size = input_size
        self.original_resnet_architecture = original_resnet_architecture
        self.return_raw_predictions = return_raw_predictions

    def get_ls_mcd_samples_baselines(self, data_loader: torch.utils.data.dataloader.DataLoader):
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
                dl_imgs_latent_mcd_samples = []
                if self.return_raw_predictions:
                    raw_predictions = []
                for i, (image, label) in enumerate(data_loader):
                    # image = image.view(1, 1, 28, 28).to(device)
                    image = image.to(self.device)
                    if self.return_raw_predictions:
                        latent_samples, raw_preds = self._get_mcd_samples_one_image_baselines(image=image)
                        dl_imgs_latent_mcd_samples.append(latent_samples)
                        raw_predictions.append(raw_preds)
                    else:
                        dl_imgs_latent_mcd_samples.append(self._get_mcd_samples_one_image_baselines(image=image))
                    # Update progress bar
                    pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)
        print("MCD N_samples: ", dl_imgs_latent_mcd_samples_t.shape[1])
        if self.return_raw_predictions:
            return dl_imgs_latent_mcd_samples_t, torch.cat(raw_predictions, dim=0)
        else:
            return dl_imgs_latent_mcd_samples_t

    def _get_mcd_samples_one_image_baselines(self, image):
        img_mcd_samples = []
        for s in range(self.mcd_nro_samples):
            pred_img = self.model(image)
            # pred = torch.argmax(pred_img, dim=1)
            latent_mcd_sample = self.hook_dropout_layer.output
            if self.layer_type == "Conv":
                if self.architecture == "small":
                    # Get image HxW mean:
                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                    # latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                    # Remove useless dimensions:
                    # latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                    latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                    latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                # Resnet 18
                else:
                    # latent_mcd_sample = dropblock_ext(latent_mcd_sample)
                    # For 2nd conv layer block of resnet 18:
                    if self.location == 2:
                        # To conserve the most info, while also aggregating: let us reshape then average
                        if self.input_size == 32:
                            if self.original_resnet_architecture:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 4, 4])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    # Perform average pooling over latent representations
                                    # For input of size 32
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=2, stride=2, padding=0
                                    )
                            # Modified Lightning arch
                            else:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 16, 16])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 8, -1)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    # Perform average pooling over latent representations
                                    # For input of size 32
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=8, stride=6, padding=2
                                    )
                        # Input size 64
                        elif self.input_size == 64:
                            assert latent_mcd_sample.shape == torch.Size([1, 128, 32, 32])
                            if self.reduction_method == "mean":
                                latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 8, -1)
                                latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                            else:
                                # For input of size 64
                                latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=16, stride=12, padding=4)
                        # Input size 128
                        else:
                            if self.original_resnet_architecture:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 16, 16])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    # Perform average pooling over latent representations
                                    # For input of size 128
                                    # latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=2, stride=2, padding=0)
                                    raise NotImplementedError
                            # Modified pytorch lightning Resnet Architecture
                            else:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 64, 64])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 8, -1)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                else:
                                    # For input of size 64
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=16, stride=12, padding=4
                                    )

                        latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    elif self.location == 1:
                        assert latent_mcd_sample.shape == torch.Size([1, 64, 32, 32])
                        # latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                        if self.reduction_method == "mean":
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                            latent_mcd_sample = latent_mcd_sample.reshape(1, 64, 16, -1)
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                            latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                        # Avg pool
                        else:
                            latent_mcd_sample = avg_pool2d(latent_mcd_sample, kernel_size=4, stride=2, padding=2)
                        latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    elif self.location == 3:
                        if self.input_size == 32:
                            if self.original_resnet_architecture:
                                assert latent_mcd_sample.shape == torch.Size([1, 256, 2, 2])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    raise NotImplementedError
                            # Modified Lightning arch
                            else:
                                assert latent_mcd_sample.shape == torch.Size([1, 256, 8, 8])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 256, 4, -1)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                                    latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=4, stride=4, padding=0
                                    )
                                latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                            # latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                        else:
                            raise NotImplementedError
                        latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    else:
                        raise NotImplementedError
                        # Get image HxW mean:
                        # latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                        # latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                        # # Remove useless dimensions:
                        # latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                        # latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                        # latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
            # FC
            else:
                # It is already a 1d tensor
                # latent_mcd_sample = dropout_ext(latent_mcd_sample)
                latent_mcd_sample = torch.squeeze(latent_mcd_sample)
            img_mcd_samples.append(latent_mcd_sample)

        if self.layer_type == "Conv":
            img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
        else:
            img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
        if self.return_raw_predictions:
            return img_mcd_samples_t, pred_img
        else:
            return img_mcd_samples_t


def get_mcd_pred_uncertainty_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader, mcd_nro_samples: int = 2):
    """
    This function calculates the predictive uncertainty, the mutual information, and returns the predictions,
    given a model, a dataloader and a number of MCD steps
    :param dnn_model: Trained model
    :type dnn_model: torch.nn.Module
    :param input_dataloader: Data Loader
    :type input_dataloader: DataLoader
    :param mcd_nro_samples: Number of samples for MCD dropout
    :type mcd_nro_samples: int
    """
    softmax_fn = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gtsrb_model.to(device)
    with torch.no_grad():
        # dl_imgs_latent_mcd_samples = []
        # dl_pred_mcd_samples = []
        img_pred_mcd_samples = []

        for i, (image, label) in enumerate(tqdm(input_dataloader)):
            image = image.to(device)

            for sample in range(mcd_nro_samples):
                pred_img = dnn_model(image)

                img_pred_mcd_samples.append(pred_img)

        img_pred_mcd_samples_t = torch.cat(img_pred_mcd_samples, dim=0)
        # ic(img_pred_mcd_samples_t.shape)

        # compute softmax output - normalized output:
        img_pred_softmax_mcd_samples_t = softmax_fn(img_pred_mcd_samples_t)

        dl_pred_mcd_samples = torch.split(img_pred_softmax_mcd_samples_t, mcd_nro_samples)
        # Get dataloader mcd predictions:
        dl_pred_mcd_samples_t = torch.stack(dl_pred_mcd_samples)

        # get predictive entropy:
        expect_preds = torch.mean(dl_pred_mcd_samples_t, dim=1)
        pred_h_t = -torch.sum((expect_preds * torch.log(expect_preds)), dim=1)
        # get expected entropy:
        preds_h = -torch.sum(dl_pred_mcd_samples_t * torch.log(dl_pred_mcd_samples_t), dim=-1)
        expected_h_preds_t = torch.mean(preds_h, dim=1)
        # get mutual information:
        mi_t = pred_h_t - expected_h_preds_t

    return dl_pred_mcd_samples_t, pred_h_t, mi_t


def get_predictive_uncertainty_score(input_samples: Tensor, mcd_nro_samples: int):
    """
    This function calculates the predictive uncertainty, the mutual information, and returns the predictions,
    given a model, a dataloader and a number of MCD steps
    :param input_samples: Already calculated outputs from a model with the given MCD steps
    :param mcd_nro_samples: Number of samples for MCD dropout
    :type mcd_nro_samples: int
    :return: predictive uncertainty, mutual information
    """
    softmax_fn = torch.nn.Softmax(dim=1)
    # compute softmax output - normalized output:
    img_pred_softmax_mcd_samples_t = softmax_fn(input_samples)

    dl_pred_mcd_samples = torch.split(img_pred_softmax_mcd_samples_t, mcd_nro_samples)
    # Get dataloader mcd predictions:
    dl_pred_mcd_samples_t = torch.stack(dl_pred_mcd_samples)

    # get predictive entropy:
    expect_preds = torch.mean(dl_pred_mcd_samples_t, dim=1)
    pred_h_t = -torch.sum((expect_preds * torch.log(expect_preds)), dim=1)
    # get expected entropy:
    preds_h = -torch.sum(dl_pred_mcd_samples_t * torch.log(dl_pred_mcd_samples_t), dim=-1)
    expected_h_preds_t = torch.mean(preds_h, dim=1)
    # get mutual information:
    mi_t = pred_h_t - expected_h_preds_t

    return pred_h_t, mi_t


def get_msp_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader):
    """
    Calculates the Maximum softmax probability score
    """
    softmax_fn = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_model.to(device)
    dl_preds_msp_scores = []

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(input_dataloader)):
            image = image.to(device)
            pred_logits = dnn_model(image)

            pred_score = torch.max(softmax_fn(pred_logits), dim=1)
            # ic(pred_score.shape)
            # get the max values:
            dl_preds_msp_scores.append(pred_score[0])

        dl_preds_msp_scores_t = torch.cat(dl_preds_msp_scores, dim=0)
        # ic(dl_preds_msp_scores_t.shape)
        # pred = np.max(softmax_fn(pred_logits).detach().cpu().numpy(), axis=1)
        dl_preds_msp_scores = dl_preds_msp_scores_t.detach().cpu().numpy()

    return dl_preds_msp_scores


def get_energy_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader):
    """
    Calculates the energy uncertainty score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_det_model.to(device)

    dl_preds_energy_scores = []

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(input_dataloader)):
            image = image.to(device)
            pred_logits = dnn_model(image)

            pred_energy_score = torch.logsumexp(pred_logits, dim=1)

            dl_preds_energy_scores.append(pred_energy_score)

        dl_preds_energy_scores_t = torch.cat(dl_preds_energy_scores, dim=0)

        dl_preds_energy_scores = dl_preds_energy_scores_t.detach().cpu().numpy()

    return dl_preds_energy_scores


class MDSPostprocessor:
    """
    Mahalanobis Distance Score uncertainty estimator class
    """

    def __init__(self, num_classes: int = 43, setup_flag: bool = False):
        """
        :param num_classes: Number of In-distribution samples
        :type num_classes: int
        :param setup_flag: Whether the postprocessor is already trained
        :type setup_flag: bool
        """
        # self.config = config
        # self.num_classes = num_classes_dict[self.config.dataset.name]
        self.num_classes = num_classes
        self.setup_flag = setup_flag

    def setup(self, dnn_model: torch.nn.Module, ind_dataloader, layer_hook):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print("\n Estimating mean and variance from training set...")
            all_feats = []
            all_labels = []
            all_preds = []
            # get features/representations:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dnn_model.to(device)
            # get features:
            with torch.no_grad():
                for i, (image, label) in enumerate(tqdm(ind_dataloader)):
                    image = image.to(device)
                    # label = label.to(device)
                    pred_logits = dnn_model(image)
                    latent_rep = torch.flatten(layer_hook.output, 1)  # latent representation sample
                    all_feats.append(latent_rep.cpu())
                    all_labels.append(deepcopy(label))
                    all_preds.append(pred_logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # compute class-conditional statistics:
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples - self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(self.class_mean)  # shape [#classes, feature dim]

            # group_lasso = sklearn.covariance.EmpiricalCovariance(
            #     assume_centered=False)

            group_lasso = EmpiricalCovariance(assume_centered=False)

            group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, dnn_model: torch.nn.Module, dataloader: DataLoader, layer_hook):
        all_preds = []
        all_conf_score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dnn_model.to(device)

        for i, (image, label) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            pred_logits = dnn_model(image)
            # latent_rep = layer_hook.output
            latent_rep = torch.flatten(layer_hook.output, 1)
            pred = pred_logits.argmax(1)

            all_preds.append(pred)

            class_scores = torch.zeros((pred_logits.shape[0], self.num_classes))
            for c in range(self.num_classes):
                tensor = latent_rep.cpu() - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(torch.matmul(tensor, self.precision), tensor.t()).diag()

            conf = torch.max(class_scores, dim=1)[0]

            all_conf_score.append(conf)

        all_preds_t = torch.cat(all_preds)
        all_conf_score_t = torch.cat(all_conf_score)

        return all_preds_t, all_conf_score_t


normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


class KNNPostprocessor:
    def __init__(self, K: int = 50, setup_flag: bool = False):
        self.K = K
        self.activation_log = None
        self.setup_flag = setup_flag
        self.index = None

    def setup(self, dnn_model: torch.nn.Module, ind_dataloader, layer_hook, get_2d_rep_mean: bool = False):
        if not self.setup_flag:
            print("\n Get latent embeddings z from training set...")
            activation_log = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dnn_model.to(device)

            with torch.no_grad():
                for i, (image, label) in enumerate(ind_dataloader):
                    image = image.to(device)
                    pred_logits = dnn_model(image)

                    latent_rep = torch.flatten(layer_hook.output, 1)  # latent representation sample
                    # ic(layer_hook.output)
                    activation_log.append(normalizer(latent_rep.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(latent_rep.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, dnn_model: torch.nn.Module, dataloader: DataLoader, layer_hook):
        all_preds = []
        all_kth_dist_score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dnn_model.to(device)

        for i, (image, label) in enumerate(dataloader):
            image = image.to(device)
            pred_logits = dnn_model(image)
            # ic(layer_hook.output)
            latent_rep = torch.flatten(layer_hook.output, 1)  # latent representation sample

            pred = torch.max(torch.softmax(pred_logits, dim=1), dim=1)
            latent_rep_normed = normalizer(latent_rep.data.cpu().numpy())

            D, _ = self.index.search(latent_rep_normed, self.K)
            kth_dist = -D[:, -1]

            all_preds.append(pred[0])
            all_kth_dist_score.append(kth_dist)

        all_preds_t = torch.cat(all_preds)
        # all_kth_dist_score_t = torch.cat(all_kth_dist_score)
        all_kth_dist_score_np = np.concatenate(all_kth_dist_score, axis=0)

        return all_preds_t, all_kth_dist_score_np

    def set_K_hyperparam(self, hyperparam: int = 50):
        self.K = hyperparam

    def get_K_hyperparam(self):
        return self.K
