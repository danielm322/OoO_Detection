from typing import Tuple
import torch
import numpy as np
from dropblock import DropBlock2D
from torch.utils.data import DataLoader
from tqdm import tqdm
from .uncertainty_estimation import Hook

dropblock_ext = DropBlock2D(drop_prob=0.4, block_size=1)


def get_msp_score_rcnn(dnn_model: torch.nn.Module, input_dataloader: DataLoader) -> np.array:
    """
    Calculates the Maximum softmax probability score from an RCNN architecture coded with the
    Detectron 2 library, where the results are the first element of the output of the network,
    and the softmax are already calculated within the scores attribute of the results.

    Args:
        dnn_model (torch.nn.Module): The RCNN model
        input_dataloader (DataLoader): The Dataloader

    Returns:
        np.array: The MSP scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_model.to(device)
    dl_preds_msp_scores = []

    with torch.no_grad():
        for i, image in enumerate(tqdm(input_dataloader, desc="Getting MSP score")):
            # image = image.to(device)
            results, _ = dnn_model(image)

            pred_scores = results.scores
            # ic(pred_score.shape)
            # get the max values:
            if len(pred_scores) > 0:
                dl_preds_msp_scores.append(pred_scores.max().reshape(1))
            else:
                dl_preds_msp_scores.append((torch.Tensor([0.0])).to(device))

        dl_preds_msp_scores_t = torch.cat(dl_preds_msp_scores, dim=0)
        # ic(dl_preds_msp_scores_t.shape)
        dl_preds_msp_scores = dl_preds_msp_scores_t.detach().cpu().numpy()

    return dl_preds_msp_scores


def get_dice_feat_mean_react_percentile_rcnn(
    dnn_model: torch.nn.Module, ind_dataloader: DataLoader, react_percentile: int = 90
) -> Tuple[np.array, float]:
    """
    Get the DICE and ReAct thresholds for sparsifying and clipping from an RCNN architecture, where
    the output has been modified to return the previous-to-last layer activations.
    Args:
        dnn_model: The RCNN model
        ind_dataloader: The Data loader
        react_percentile: Desired percentile for ReAct

    Returns:
        Tuple[np.array, float]: The DICE expected values, and the ReAct threshold
    """
    feat_log = []
    dnn_model.model.eval()
    assert dnn_model.dice_react_precompute
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs in tqdm(ind_dataloader, desc="Setting up DICE/ReAct"):
        outputs = dnn_model(inputs)
        out = outputs.mean(0)
        out = out.view(1, -1)
        # score = dnn_model.fc(out)
        feat_log.append(out.data.cpu().numpy())
    feat_log_array = np.array(feat_log).squeeze()
    return feat_log_array.mean(0), np.percentile(feat_log_array, react_percentile)


def get_energy_score_rcnn(dnn_model: torch.nn.Module, input_dataloader: DataLoader):
    """
    Calculates the energy uncertainty score from an RCNN architecture where the output has been
    modified to return the raw activations before NMS alongside the normal (NMS filtered) ones.
    Args:
        dnn_model: The RCNN model
        input_dataloader: The Data loader

    Returns:
        Tuple[np.array, np.array]: Energy scores from the Raw and the filtered outputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gtsrb_det_model.to(device)

    # Here we take the enrgy as a mean of the whole 1000 proposals
    raw_preds_energy_scores = []
    # Here we take the enrgy as a mean of the filtered detections after NMS
    filtered_preds_energy_scores = []

    with torch.no_grad():
        for i, image in enumerate(tqdm(input_dataloader, desc="Getting energy score")):
            results, box_cls = dnn_model(image)
            # Raw energy
            raw_energy_score = torch.logsumexp(box_cls[:, :-1], dim=1)
            raw_preds_energy_scores.append(raw_energy_score.mean().reshape(1))
            # Filtered energy
            filtered_energy_score = torch.logsumexp(results.inter_feat[:, :-1], dim=1)
            filtered_preds_energy_scores.append(filtered_energy_score.mean().reshape(1))

        raw_preds_energy_scores_t = torch.cat(raw_preds_energy_scores, dim=0)
        raw_preds_energy_scores = raw_preds_energy_scores_t.detach().cpu().numpy()
        filtered_preds_energy_scores_t = torch.cat(filtered_preds_energy_scores, dim=0)
        filtered_preds_energy_scores = filtered_preds_energy_scores_t.detach().cpu().numpy()

    return raw_preds_energy_scores, filtered_preds_energy_scores


# Get latent space Monte Carlo Dropout samples
def get_ls_mcd_samples_rcnn(
    model: torch.nn.Module,
    data_loader: torch.utils.data.dataloader.DataLoader,
    mcd_nro_samples: int,
    hook_dropout_layer: Hook,
    layer_type: str,
    return_raw_predictions: bool,
) -> torch.tensor:
    """
    Get Monte-Carlo Dropout samples from RCNN's Dropout or Dropblock Layer.
    For this function to work on the RPN, it must have defined a list called
    'rpn_intermediate_output' that collects the intermediate representations at inference time
    i.e. the hook is not enough since it only captures one output per layer and this layer
    outputs five elements. A modification in the RPN is therefore needed.
    An example below::

        def forward(self, features: List[torch.Tensor]):
            self.rpn_intermediate_output.clear()
            pred_objectness_logits = []
            pred_anchor_deltas = []
            for x in features:
                t = self.dropblock(self.conv(x))
                # Normal NN operations
                pred_objectness_logits.append(self.objectness_logits(t))
                pred_anchor_deltas.append(self.anchor_deltas(t))
                # Hook operations to catch RPN intermediate output
                self.rpn_intermediate_output.append(t)
            return pred_objectness_logits, pred_anchor_deltas

    Args:
        model: Torch model
        data_loader: Input samples (torch) DataLoader
        mcd_nro_samples: Number of Monte-Carlo Samples
        hook_dropout_layer: Hook at the Dropout Layer from the Neural Network Module
        layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or
            Conv (Convolutional)
        return_raw_predictions: Returns the raw logits output

    Returns:
        Monte-Carlo Dropout samples for the input dataloader
    """
    assert layer_type in ("FC", "Conv", "RPN", "backbone"), (
        "Layer type must be either 'FC','backbone', 'RPN' or 'Conv'"
    )
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
            dl_imgs_latent_mcd_samples = []
            if return_raw_predictions:
                raw_predictions = []
            for i, image in enumerate(data_loader):
                img_mcd_samples = []
                for s in range(mcd_nro_samples):
                    instances, _ = model(image)
                    if return_raw_predictions:
                        raw_predictions.append(instances.inter_feat[:, :-1].mean(0))
                    # pred = torch.argmax(pred_img, dim=1)
                    latent_mcd_sample = hook_dropout_layer.output
                    if layer_type == "Conv":
                        # Get image HxW mean:
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=2, keepdim=True)
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=3, keepdim=True)
                        # Remove useless dimensions:
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=3)
                        latent_mcd_sample = torch.squeeze(latent_mcd_sample, dim=2)
                    elif layer_type == "RPN":
                        latent_mcd_sample = (
                            model.model.proposal_generator.rpn_head.rpn_intermediate_output
                        )
                        for i in range(len(latent_mcd_sample)):
                            latent_mcd_sample[i] = torch.mean(
                                latent_mcd_sample[i], dim=2, keepdim=True
                            )
                            latent_mcd_sample[i] = torch.mean(
                                latent_mcd_sample[i], dim=3, keepdim=True
                            )
                            # Remove useless dimensions:
                            latent_mcd_sample[i] = torch.squeeze(latent_mcd_sample[i])
                        latent_mcd_sample = torch.cat(latent_mcd_sample, dim=0)
                    elif layer_type == "backbone":
                        # Apply dropblock
                        for k, v in latent_mcd_sample.items():
                            latent_mcd_sample[k] = dropblock_ext(v)
                            # Get image HxW mean:
                            latent_mcd_sample[k] = torch.mean(
                                latent_mcd_sample[k], dim=2, keepdim=True
                            )
                            latent_mcd_sample[k] = torch.mean(
                                latent_mcd_sample[k], dim=3, keepdim=True
                            )
                            # Remove useless dimensions:
                            latent_mcd_sample[k] = torch.squeeze(latent_mcd_sample[k])
                        latent_mcd_sample = torch.cat(list(latent_mcd_sample.values()), dim=0)
                    # FC
                    else:
                        # Aggregate the second dimension (dim 1) to keep the proposed boxes dim
                        latent_mcd_sample = torch.mean(latent_mcd_sample, dim=1)
                    if (
                        layer_type == "FC" and latent_mcd_sample.shape[0] == 1000
                    ) or layer_type == "RPN":
                        img_mcd_samples.append(latent_mcd_sample)
                    elif layer_type == "FC" and latent_mcd_sample.shape[0] != 1000:
                        pass
                    else:
                        raise NotImplementedError
                if layer_type == "Conv":
                    img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
                    dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                else:
                    if (
                        layer_type == "FC" and latent_mcd_sample.shape[0] == 1000
                    ) or layer_type == "RPN":
                        img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
                        dl_imgs_latent_mcd_samples.append(img_mcd_samples_t)
                    elif layer_type == "FC" and latent_mcd_sample.shape[0] != 1000:
                        print(f"Omitted image: {image[0]['image_id']}")
                    else:
                        raise NotImplementedError

                # Update progress bar
                pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)

    if return_raw_predictions:
        return dl_imgs_latent_mcd_samples_t, torch.stack(raw_predictions, dim=0)
    else:
        return dl_imgs_latent_mcd_samples_t
