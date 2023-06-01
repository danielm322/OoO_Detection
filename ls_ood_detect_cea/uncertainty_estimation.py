from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from dropblock import DropBlock2D
import pytorch_lightning as pl
from entropy_estimators import continuous
from icecream import ic
from tqdm import tqdm


def get_ls_samples_priornet(prob_module: pl.LightningModule, dataloader: DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def deeplabv3p_get_ls_mcd_samples(model_module: pl.LightningModule,
                                  dataloader: DataLoader,
                                  mcd_nro_samples: int,
                                  hook_dropout_layer: Hook) -> Tensor:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def get_latent_representation_mcd_samples(dnn_model: torch.nn.Module,
                                          dataloader: DataLoader,
                                          mcd_nro_samples: int,
                                          layer_hook: Hook,
                                          layer_type: str) -> Tensor:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
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


def probunet_get_ls_mcd_samples(model_module: pl.LightningModule,
                                dataloader: DataLoader,
                                mcd_nro_samples: int,
                                hook_dropout_layer: Hook) -> Tensor:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def get_dl_h_z(dl_z_samples: Tensor, mcd_samples_nro: int = 32) -> Tuple[np.ndarray, np.ndarray]:
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
    dl_h_mvn_z_samples_ls = [continuous.get_h(s, k=5, norm='max', min_dist=1e-5) for s in z_samples_np_ls]
    dl_h_mvn_z_samples_np = np.array(dl_h_mvn_z_samples_ls)
    dl_h_mvn_z_samples_np = np.expand_dims(dl_h_mvn_z_samples_np, axis=1)
    # ic(dl_h_mvn_z_samples_np.shape)
    # Get dataloader entropy $h(z_i)$ for each value of Z, from mcd_samples
    dl_h_z_samples = []
    for input_mcd_samples in tqdm(z_samples_np_ls, desc='Calculating entropy'):
        h_z_batch = []
        for z_val_i in range(input_mcd_samples.shape[1]):
            # h_z_i = continuous.get_h(input_mcd_samples[:, z_val_i], k=5)  # old
            h_z_i = continuous.get_h(input_mcd_samples[:, z_val_i], k=5, norm='max', min_dist=1e-5)
            h_z_batch.append(h_z_i)
        h_z_batch_np = np.asarray(h_z_batch)
        dl_h_z_samples.append(h_z_batch_np)
    dl_h_z_samples_np = np.asarray(dl_h_z_samples)
    # ic(dl_h_z_samples_np.shape)
    return dl_h_mvn_z_samples_np, dl_h_z_samples_np


def probunet_apply_dropout(m):
    if type(m) == torch.nn.Dropout or type(m) == DropBlock2D:
        m.train()


def deeplabv3p_apply_dropout(m):
    if type(m) == torch.nn.Dropout or type(m) == DropBlock2D:
        m.train()

