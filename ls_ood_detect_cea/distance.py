from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import pytorch_lightning as pl
from torch.distributions import Normal, Independent
import numpy as np
from icecream import ic


def mahalanobis_distance(x: Tensor, mean_ref_dist: Tensor, cov_ref_dist: Tensor) -> Tensor:
    """
    Calculates the Mahalanobis distance between a sample 'x', and a multivariate normal reference distribution
    with mean and covariance.

    :param x: Sample
    :type x: Tensor
    :param mean_ref_dist: Mean reference Multivariate Normal Distribution
    :type mean_ref_dist: Tensor
    :param cov_ref_dist: Covariance reference Multivariate Normal Distribution
    :type cov_ref_dist: Tensor
    :return: Mahalanobis Distance of sample 'x' to the reference distribution
    :rtype: Tensor
    """
    delta = x - mean_ref_dist
    cov_inv_ref_dist = torch.inverse(cov_ref_dist)
    m = torch.sqrt(torch.mm(torch.mm(delta, cov_inv_ref_dist), delta.t()))
    return torch.diag(m)


def mahalanobis_distance_to_ref_pdf(prob_module: pl.LightningModule,
                                    dataloader: DataLoader,
                                    ref_pdf: MultivariateNormal) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the Mahalanobis Distance to a Ref. Probability Distribution, from each batch (sample) in the dataloader

    :param prob_module: Probabilistic NN Module, (e.g., Probabilistic U-Net Module)
    :type prob_module: LightningModule
    :param dataloader: PyTorch Dataloader
    :type dataloader: Dataloader
    :param ref_pdf: Reference Probability Density Function
    :type ref_pdf: MultivariateNormal
    :return dl_m_dist_t: Dataloader samples  Mahalanobis distances to reference PDF
    :rtype dl_m_dist_t: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ic(device)
    ref_pdf_mean = ref_pdf.loc
    ref_pdf_cov = ref_pdf.covariance_matrix
    ref_pdf_cov_inv = torch.linalg.inv(ref_pdf_cov)

    with torch.no_grad():
        dl_mean_samples = []
        dl_var_samples = []
        dl_z_samples = []
        dl_m_dist = []
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
            # append distances
            m_dist_i = mahalanobis_distance(z_i, ref_pdf_mean, ref_pdf_cov_inv)
            dl_m_dist.append(m_dist_i)
            # append gt labels
            dl_labels.append(b_sample_labels)

        dl_m_dist_t = torch.cat(dl_m_dist, dim=0)
        dl_labels_t = torch.cat(dl_labels, dim=0)
        dl_mean_samples_t = torch.cat(dl_mean_samples, dim=0)
        dl_var_samples_t = torch.cat(dl_var_samples, dim=0)
        dl_z_samples_t = torch.cat(dl_z_samples, dim=0)

    return dl_m_dist_t, dl_labels_t, dl_z_samples_t, dl_mean_samples_t, dl_var_samples_t
