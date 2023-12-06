# (c) 2023, CEA LIST
#
# All rights reserved.
# SPDX-License-Identifier: MIT
#
# Contributors
#    Based on https://github.com/fregu856/deeplabv3
#    Fabio Arnez, probabilistic adaptation

from typing import Tuple, List
from typing import Tuple, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import avg_pool2d, adaptive_avg_pool2d
from dropblock import DropBlock2D
import pytorch_lightning as pl
from entropy_estimators import continuous
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from copy import deepcopy
import faiss
from tqdm.contrib.concurrent import process_map


class Hook:
    """
    This class will catch the input and output of any torch layer during forward/backward pass.

    Args:
        module (torch.nn.Module): Layer block from Neural Network Module
        backward (bool): backward-poss hook
    """

    def __init__(self, module: torch.nn.Module, backward: bool = False):
        """
        This class will catch the input and output of any torch layer during forward/backward pass.

        Args:
            module (torch.nn.Module): Layer block from Neural Network Module
            backward (bool): backward-poss hook
        """
        assert isinstance(module, torch.nn.Module), "module must be a pytorch module"
        assert isinstance(backward, bool), "backward must be a boolean"
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

    Args:
        model_module (pl.LightningModule): Deeplabv3+ Neural Network Lightning Module
        dataloader (DataLoader): Input samples (torch) Dataloader
        mcd_nro_samples (int): Number of Monte-Carlo Samples
        hook_dropout_layer (Hook): Hook at the Dropout Layer from the Neural Network Module

    Returns:
        (Tensor): Monte-Carlo Dropout samples for the input dataloader
    """
    assert isinstance(model_module, torch.nn.Module), "model_module must be a pytorch model"
    assert isinstance(dataloader, DataLoader), "dataloader must be a DataLoader"
    assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
    assert isinstance(hook_dropout_layer, Hook), "hook_dropout_layer must be an Hook"
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
    Get latent representations Monte-Carlo samples from DNN using a layer hook

    Args:
        dnn_model: Neural Network Torch or Lightning Module
        dataloader: Input samples (torch) Data loader
        mcd_nro_samples: Number of Monte-Carlo Samples
        layer_hook: DNN layer hook
        layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected)
        or Conv (Convolutional)

    Returns:
        Input dataloader latent representations MC samples tensor
    """
    assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
    assert isinstance(dataloader, DataLoader), "dataloader must be a DataLoader"
    assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
    assert isinstance(layer_hook, Hook), "layer_hook must be an Hook"
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
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes
                        # dimension
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


def single_image_entropy_calculation(sample: np.ndarray, neighbors: int) -> np.ndarray:
    """
    Function used to calculate the entropy values of a single image. Used to calculate entropy
    in parallel

    Args:
        sample: MCD samples for a single image
        neighbors: Number of neighbors to perform calculation. By default 5 seems ok, but should
            be n-1 if n<=5

    Returns:
        Entropy of the activations for a single image
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

    Args:
        dl_z_samples: Dataloader Z Samples
        mcd_samples_nro: Number of monte carlo dropout samples
        parallel_run: Optionally try to perform entropy calculations in parallel

    Returns:
        Latent vector multivariate normal entropy $h(Z)$, Latent vector value entropy $h(z_i)$
    """
    assert isinstance(dl_z_samples, Tensor), "dl_z_samples must be a Tensor"
    assert isinstance(mcd_samples_nro, int), "mcd_samples_nro must be an integer"
    assert isinstance(parallel_run, bool), "parallel_run must be a boolean"
    # Get dataloader mvn h(z), from mcd_samples
    z_samples_ls = [i for i in dl_z_samples.split(mcd_samples_nro)]
    # ic(z_samples_ls[0].shape)
    z_samples_np_ls = [t.cpu().numpy() for t in z_samples_ls]
    # ic(z_samples_np_ls[0].shape)
    # dl_h_mvn_z_samples_ls = [continuous.get_h_mvn(s) for s in z_samples_np_ls]
    # Choose correctly the number of neighbors for the entropy calculations:
    # It has to be smaller than the mcd_samples_nro by at least 1
    k_neighbors = 5 if mcd_samples_nro > 5 else mcd_samples_nro - 1
    dl_h_mvn_z_samples_ls = [
        continuous.get_h(s, k=k_neighbors, norm="max", min_dist=1e-5) for s in z_samples_np_ls
    ]
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
                h_z_i = continuous.get_h(
                    input_mcd_samples[:, z_val_i], k=k_neighbors, norm="max", min_dist=1e-5
                )
                h_z_batch.append(h_z_i)
            h_z_batch_np = np.asarray(h_z_batch)
            dl_h_z_samples.append(h_z_batch_np)
    else:
        dl_h_z_samples = process_map(
            single_image_entropy_calculation,
            z_samples_np_ls,
            [k_neighbors] * len(z_samples_np_ls),
            chunksize=1,
        )
    dl_h_z_samples_np = np.asarray(dl_h_z_samples)
    # ic(dl_h_z_samples_np.shape)
    return dl_h_mvn_z_samples_np, dl_h_z_samples_np


def probunet_apply_dropout(m):
    """
    Activate Dropout or Dropblock layers.
    Args:
        m: Pytorch module
    """
    if type(m) == torch.nn.Dropout or type(m) == DropBlock2D:
        m.train()


def deeplabv3p_apply_dropout(m):
    """
    Activate Dropout or Dropblock layers.
    Args:
        m: Pytorch module
    """
    if type(m) == torch.nn.Dropout or type(m) == DropBlock2D:
        m.train()


def get_mean_or_fullmean_ls_sample(latent_sample: Tensor, method: str):
    assert method in ("mean", "fullmean")
    if method == "mean":
        latent_sample = torch.mean(latent_sample, dim=3, keepdim=True)
        latent_sample = torch.squeeze(latent_sample)
    # fullmean
    else:
        latent_sample = torch.mean(latent_sample, dim=3, keepdim=True)
        latent_sample = torch.mean(latent_sample, dim=2, keepdim=True)
        latent_sample = torch.squeeze(latent_sample)
    return latent_sample


class MCDSamplesExtractor:
    """
    Class to get Monte-Carlo samples from any torch model Dropout or Dropblock Layer

    Args:
        model: Torch or Lightning model
        mcd_nro_samples: Number of Monte-Carlo Samples
        hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
        layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or
            Conv (Convolutional)
        device: CUDA or CPU device
        architecture: The model architecture: either small or resnet
        location: Location of the hook. This can be useful to select different latent sample
            catching layers
        reduction_method: Whether to use fullmean, mean, or avgpool to reduce dimensionality
            of hooked representation
        input_size: Image size (we assume squared images)
        original_resnet_architecture: Either original or modified Lightning architecture
        return_raw_predictions: Return or not network outputs

    Returns:
        Monte-Carlo Dropout samples for the input dataloader
    """

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
        Class to get Monte-Carlo samples from any torch model Dropout or Dropblock Layer

        Args:
            model: Torch or Lightning model
            mcd_nro_samples: Number of Monte-Carlo Samples
            hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
            layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or
                Conv (Convolutional)
            device: CUDA or CPU device
            architecture: The model architecture: either small or resnet
            location: Location of the hook. This can be useful to select different latent sample
                catching layers
            reduction_method: Whether to use fullmean, mean, or avgpool to reduce dimensionality
                of hooked representation
            input_size: Image size (we assume squared images)
            original_resnet_architecture: Either original or modified Lightning architecture
            return_raw_predictions: Return or not network outputs

        Returns:
            Monte-Carlo Dropout samples for the input dataloader
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
        assert isinstance(model, torch.nn.Module), "model must be a pytorch model"
        assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
        assert isinstance(hook_dropout_layer, Hook), "hook_dropout_layer must be an Hook"
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

    def get_ls_mcd_samples_baselines(
        self, data_loader: DataLoader
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Perform the Monte Carlo Dropout inference given a dataloader
        Args:
            data_loader: DataLoader

        Returns:
            Latent MCD samples and optionally the raw inference results
        """
        assert isinstance(data_loader, DataLoader)
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
                dl_imgs_latent_mcd_samples = []
                if self.return_raw_predictions:
                    raw_predictions = []
                for i, (image, label) in enumerate(data_loader):
                    # image = image.view(1, 1, 28, 28).to(device)
                    image = image.to(self.device)
                    if self.return_raw_predictions:
                        latent_samples, raw_preds = self._get_mcd_samples_one_image_baselines(
                            image=image
                        )
                        dl_imgs_latent_mcd_samples.append(latent_samples)
                        raw_predictions.extend(raw_preds)
                    else:
                        dl_imgs_latent_mcd_samples.append(
                            self._get_mcd_samples_one_image_baselines(image=image)
                        )
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
        if self.return_raw_predictions:
            raw_predictions = []
        for s in range(self.mcd_nro_samples):
            pred_img = self.model(image)
            if self.return_raw_predictions:
                raw_predictions.append(pred_img)
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
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=2, keepdim=True
                                    )
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
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 8, -1)
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
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
                            if self.original_resnet_architecture:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 8, 8])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=2, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    # Perform average pooling over latent representations
                                    # For input of size 32
                                    raise NotImplementedError
                            # Modified Lightning arch
                            else:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 32, 32])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 8, -1)
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                else:
                                    # For input of size 64
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=16, stride=12, padding=4
                                    )
                        # Input size 128
                        else:
                            if self.original_resnet_architecture:
                                assert latent_mcd_sample.shape == torch.Size(
                                    [1, 128, 16, 16]
                                ), f"got {latent_mcd_sample.shape}"
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=2, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    # Perform average pooling over latent representations
                                    # For input of size 128
                                    # latent_mcd_sample = avg_pool2d(
                                    # latent_mcd_sample, kernel_size=2, stride=2, padding=0
                                    # )
                                    raise NotImplementedError
                            # Modified pytorch lightning Resnet Architecture
                            else:
                                assert latent_mcd_sample.shape == torch.Size([1, 128, 64, 64])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 8, -1)
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                else:
                                    # For input of size 64
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=16, stride=12, padding=4
                                    )

                        latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    elif self.location == 1:
                        assert self.input_size == 128, f"Input size: {self.input_size}"
                        assert latent_mcd_sample.shape == torch.Size([1, 64, 32, 32])
                        # latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                        if self.reduction_method == "mean":
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                            latent_mcd_sample = latent_mcd_sample.reshape(1, 64, 16, -1)
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                            latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                        elif self.reduction_method == "fullmean":
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                            latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                            latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                        # Avg pool
                        else:
                            latent_mcd_sample = avg_pool2d(
                                latent_mcd_sample, kernel_size=4, stride=2, padding=2
                            )
                        latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    elif self.location == 3:
                        if self.input_size == 32:
                            if self.original_resnet_architecture:
                                assert latent_mcd_sample.shape == torch.Size([1, 256, 2, 2])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=2, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    raise NotImplementedError
                            # Modified Lightning arch
                            else:
                                assert latent_mcd_sample.shape == torch.Size([1, 256, 8, 8])
                                if self.reduction_method == "mean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = latent_mcd_sample.reshape(1, 256, 4, -1)
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                elif self.reduction_method == "fullmean":
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=3, keepdim=True
                                    )
                                    latent_mcd_sample = torch.mean(
                                        latent_mcd_sample, dim=2, keepdim=True
                                    )
                                    latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                                # Avg pool
                                else:
                                    latent_mcd_sample = avg_pool2d(
                                        latent_mcd_sample, kernel_size=4, stride=4, padding=0
                                    )
                                latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                            # latent_mcd_sample = latent_mcd_sample.reshape(1, 128, 4, -1)
                        elif self.input_size == 128:
                            assert self.original_resnet_architecture, "Not implemented otherwise"
                            assert latent_mcd_sample.shape == torch.Size(
                                [1, 256, 8, 8]
                            ), f"got {latent_mcd_sample.shape}"
                            if self.reduction_method == "mean":
                                latent_mcd_sample = torch.mean(
                                    latent_mcd_sample, dim=3, keepdim=True
                                )
                                latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                            elif self.reduction_method == "fullmean":
                                latent_mcd_sample = torch.mean(
                                    latent_mcd_sample, dim=3, keepdim=True
                                )
                                latent_mcd_sample = torch.mean(
                                    latent_mcd_sample, dim=2, keepdim=True
                                )
                                latent_mcd_sample = torch.squeeze(latent_mcd_sample)
                            # Avg pool
                            else:
                                raise NotImplementedError
                        else:
                            raise NotImplementedError
                        latent_mcd_sample = latent_mcd_sample.reshape(1, -1)
                    else:
                        raise NotImplementedError
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
            return img_mcd_samples_t, raw_predictions
        else:
            return img_mcd_samples_t


def get_mcd_pred_uncertainty_score(
    dnn_model: torch.nn.Module, input_dataloader: DataLoader, mcd_nro_samples: int = 2
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    This function performs inference and calculates the predictive uncertainty, the mutual
    information, and returns the predictions, given a model, a dataloader and a number of MCD steps.

    Args:
        dnn_model: Trained model
        input_dataloader: Data Loader
        mcd_nro_samples: Number of samples for MCD dropout

    Returns:
        MCD samples, predictive entropy and mutual information scores
    """
    assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
    assert isinstance(input_dataloader, DataLoader), "input_dataloader must be a DataLoader"
    assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
    softmax_fn = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gtsrb_model.to(device)
    with torch.no_grad():
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


def get_predictive_uncertainty_score(
    input_samples: Tensor, mcd_nro_samples: int
) -> Tuple[Tensor, Tensor]:
    """
    This function calculates the predictive uncertainty and the mutual information given some
    already calculated activations for a number of MCD steps.

    Args:
        input_samples: Already calculated outputs from a model with the given MCD steps
        mcd_nro_samples: Number of samples for MCD dropout

    Returns:
        Predictive uncertainty, mutual information
    """
    # Check correct dimensions
    assert isinstance(input_samples, Tensor), "input_samples must be a pytorch Tensor"
    assert input_samples.shape[0] % mcd_nro_samples == 0, (
        "Input tensor first dimension must be " "divisible by the mcd_nro_samples"
    )
    assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
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


def get_msp_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader) -> np.ndarray:
    """
    Calculates the Maximum softmax probability score given a model and a dataloader

    Args:
        dnn_model: Trained torch or lightning model
        input_dataloader: Dataloader

    Returns:
        MSP scores
    """
    assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
    assert isinstance(input_dataloader, DataLoader), "input_dataloader must be a DataLoader"

    softmax_fn = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_model.to(device)
    dl_preds_msp_scores = []

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(input_dataloader, desc="Getting MSP score")):
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


def get_energy_score(dnn_model: torch.nn.Module, input_dataloader: DataLoader) -> np.ndarray:
    """
    Calculates the energy uncertainty score

    Args:
        dnn_model: Trained torch or lightning model
        input_dataloader: Dataloader

    Returns:
        Energy scores
    """
    assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
    assert isinstance(input_dataloader, DataLoader), "input_dataloader must be a DataLoader"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_det_model.to(device)

    dl_preds_energy_scores = []

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(input_dataloader, desc="Getting energy score")):
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

    Args:
        num_classes: Number of In-distribution samples
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, num_classes: int = 43, setup_flag: bool = False):
        """
        Mahalanobis Distance Score uncertainty estimator class

        Args:
            num_classes: Number of In-distribution samples
            setup_flag: Whether the postprocessor is already trained
        """
        assert isinstance(num_classes, int), "num_classes must be an integer"
        assert isinstance(setup_flag, bool), "setup_flag must be a boolean"

        # self.config = config
        # self.num_classes = num_classes_dict[self.config.dataset.name]
        self.num_classes = num_classes
        self.setup_flag = setup_flag

    def setup(
        self, dnn_model: torch.nn.Module, ind_dataloader: DataLoader, layer_hook: Hook
    ) -> None:
        """
        Estimate the parameters of a multivariate normal distribution from a set of data

        Args:
            dnn_model: Trained torch or lightning model
            ind_dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        """
        assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
        assert isinstance(ind_dataloader, DataLoader), "input_dataloader must be a DataLoader"
        assert isinstance(layer_hook, Hook), "layer_hook must be a Hook"

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
                for i, (image, label) in enumerate(tqdm(ind_dataloader, desc="Setting MDist")):
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
    def postprocess(
        self, dnn_model: torch.nn.Module, dataloader: DataLoader, layer_hook: Hook
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform inference with the set up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            dnn_model: Trained torch or lightning model
            dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        Returns:
            (tuple): Model predictions and confidence scores
        """
        assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
        assert isinstance(dataloader, DataLoader), "dataloader must be a DataLoader"
        assert isinstance(layer_hook, Hook), "layer_hook must be a Hook"

        all_preds = []
        all_conf_score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dnn_model.to(device)

        for i, (image, label) in enumerate(tqdm(dataloader, desc="Calculating MDist")):
            image = image.to(device)
            pred_logits = dnn_model(image)
            # latent_rep = layer_hook.output
            latent_rep = torch.flatten(layer_hook.output, 1)
            pred = pred_logits.argmax(1)

            all_preds.append(pred)

            class_scores = torch.zeros((pred_logits.shape[0], self.num_classes))
            for c in range(self.num_classes):
                tensor = latent_rep.cpu() - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, self.precision), tensor.t()
                ).diag()

            conf = torch.max(class_scores, dim=1)[0]

            all_conf_score.append(conf)

        all_preds_t = torch.cat(all_preds)
        all_conf_score_t = torch.cat(all_conf_score)

        return all_preds_t, all_conf_score_t


normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)


class KNNPostprocessor:
    """
    kNN Distance Score uncertainty estimator class

    Args:
        K: Number of neighbors for calculations
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, K: int = 50, setup_flag: bool = False):
        """
        kNN Distance Score uncertainty estimator class

        Args:
            K: Number of neighbors for calculations
            setup_flag: Whether the postprocessor is already trained
        """
        assert isinstance(K, int), "K must be an integer"
        assert isinstance(setup_flag, bool), "setup_flag must be a boolean"
        self.K = K
        self.activation_log = None
        self.setup_flag = setup_flag
        self.index = None

    def setup(
        self, dnn_model: torch.nn.Module, ind_dataloader: DataLoader, layer_hook: Hook
    ) -> None:
        """
        Estimate the parameters of a kNN estimator

        Args:
            dnn_model: Trained torch or lightning model
            ind_dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        """
        assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
        assert isinstance(ind_dataloader, DataLoader), "ind_dataloader must be a DataLoader"
        assert isinstance(layer_hook, Hook), "layer_hook must be a Hook"

        if not self.setup_flag:
            print("\n Get latent embeddings z from training set...")
            activation_log = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dnn_model.to(device)

            with torch.no_grad():
                for i, (image, label) in enumerate(tqdm(ind_dataloader, desc="Setting kNN")):
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
    def postprocess(
        self, dnn_model: torch.nn.Module, dataloader: DataLoader, layer_hook: Hook
    ) -> Tuple[Tensor, np.ndarray]:
        """
        Perform inference with the set-up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            dnn_model: Trained torch or lightning model
            dataloader: Dataloader
            layer_hook: Hook to the layer to take samples from

        Returns:
            (tuple): Model predictions and confidence scores
        """
        assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
        assert isinstance(dataloader, DataLoader), "dataloader must be a DataLoader"
        assert isinstance(layer_hook, Hook), "layer_hook must be a Hook"

        all_preds = []
        all_kth_dist_score = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dnn_model.to(device)

        for i, (image, label) in enumerate(tqdm(dataloader, desc="Calculating kNN scores")):
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


class LaREMPostprocessor:
    """
    LaREM Distance Score uncertainty estimator class

    Args:
        setup_flag: Whether the postprocessor is already trained
    """

    def __init__(self, setup_flag: bool = False):
        """
        LaREM Distance Score uncertainty estimator class

        Args:
            setup_flag: Whether the postprocessor is already trained
        """
        assert isinstance(setup_flag, bool), "setup_flag must be a boolean"
        self.setup_flag = setup_flag
        self.feats_mean = None
        self.precision = None

    def setup(self, ind_feats: np.ndarray):
        """
        Estimate the parameters of a multivariate normal distribution from a set of data

        Args:
            ind_feats: InD features to estimate the distribution

        """
        assert isinstance(ind_feats, np.ndarray), "ind_feats must be a numpy array"
        assert ind_feats.ndim == 2, "ind_feats must be 2 dimensional"
        if not self.setup_flag:
            # estimate mean and variance from training set
            self.feats_mean = np.mean(ind_feats, 0, keepdims=True)

            self.centered_data = ind_feats - self.feats_mean

            group_lasso = EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(self.centered_data)

            self.precision = group_lasso.precision_

            self.setup_flag = True
            # we need to use:
            # self.feats_mean & self.precision
        else:
            pass

    def postprocess(self, ood_feats: np.ndarray):
        """
        Perform inference with the set up estimator, i.e. for each sample in the Data Loader
        estimate if it belongs to the InD distribution

        Args:
            ood_feats: Features (either InD or OoD) to estimate if they belong to the InD
                distribution

        Returns:
            (tuple): Confidence scores
        """
        assert isinstance(ood_feats, np.ndarray), "ood_feats must be a numpy array"
        assert ood_feats.ndim == 2, "ood_feats must be 2 dimensional"
        diff = ood_feats - self.feats_mean
        conf_score = -np.diag(np.matmul(np.matmul(diff, self.precision), np.transpose(diff)))

        return conf_score


def get_dice_feat_mean_react_percentile(
    dnn_model: torch.nn.Module, ind_dataloader: DataLoader, react_percentile: int = 90
) -> Tuple[np.ndarray, float]:
    """
    Get the DICE and ReAct thresholds for sparsifying and clipping from a given model.

    Args:
        dnn_model: The RCNN model
        ind_dataloader: The Data loader
        react_percentile: Desired percentile for ReAct

    Returns:
        Tuple[np.ndarray, float]: The DICE expected values, and the ReAct threshold
    """
    assert isinstance(dnn_model, torch.nn.Module), "dnn_model must be a pytorch model"
    assert isinstance(ind_dataloader, DataLoader), "ind_dataloader must be a DataLoader"
    assert isinstance(react_percentile, int), "react_percentile must be an integer"
    assert 0 < react_percentile < 100, "react_percentile must be greater than 0 and less than 100"
    feat_log = []
    dnn_model.eval()
    assert dnn_model.dice_precompute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, targets in tqdm(ind_dataloader, desc="Setting up DICE/ReAct"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = dnn_model(inputs)
        out = adaptive_avg_pool2d(outputs, 1)
        out = out.view(out.size(0), -1)
        # score = dnn_model.fc(out)
        feat_log.append(out.data.cpu().numpy())
    feat_log_array = np.array(feat_log).squeeze()
    return feat_log_array.mean(0), np.percentile(feat_log_array, react_percentile)


"""
DICE Code taken from https://github.com/deeplearning-wisc/dice/blob/master/models/route.py
All credits to authors
"""


class RouteDICE(torch.nn.Linear):
    """
    Class to replace the penultimate fully connected layer of a network in order to use the
    DICE method
    Args:
        in_features: Dimension of the input vector
        out_features: Dimension of the output vector
        bias: Bias for the Linear layer
        p: Percentile for sparsifying
        conv1x1: Whether using a 1x1 conv layer
        info: The previously calculated expected values
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        p: int = 90,
        conv1x1: bool = False,
        info: Union[None, np.ndarray] = None,
    ):
        """
        Class to replace the penultimate fully connected layer of a network in order to use the
        DICE method
        Args:
            in_features: Dimension of the input vector
            out_features: Dimension of the output vector
            bias: Bias for the Linear layer
            p: Percentile for sparsifying
            conv1x1: Whether using a 1x1 conv layer
            info: The previously calculated expected values
        """
        assert isinstance(in_features, int), "in_features must be an integer"
        assert isinstance(out_features, int), "out_features must be an integer"
        assert isinstance(bias, bool), "bias must be a boolean"
        assert isinstance(p, int), "p must be an integer"
        assert 0 < p < 100, "p must be greater than 0 and less than 100"
        assert isinstance(conv1x1, bool), "conv1x1 must be a boolean"
        if info is not None:
            assert isinstance(info, np.ndarray), "info must be a numpy array or None"

        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        # self.contrib = np.abs(self.contrib)
        # self.contrib = np.random.rand(*self.contrib.shape)
        # self.contrib = self.info[None, :]
        # self.contrib = np.random.rand(*self.info[None, :].shape)
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out
