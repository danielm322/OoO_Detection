from typing import Tuple, Union, List, Any
import torch
from torch.nn import Dropout
from torch.utils.data import DataLoader
from torch import Tensor
from dropblock import DropBlock2D
from tqdm import tqdm
from .uncertainty_estimation import Hook, get_mean_or_fullmean_ls_sample


class NonInvasiveMCDSamplesExtractorYolov8:
    """
    Class to get Monte-Carlo samples from any Yolo v8 model layer different from a Dropout or
    Dropblock, so absolutely no architecture modification is needed

    Args:
        model: Torch or Lightning model
        mcd_nro_samples: Number of Monte-Carlo Samples
        hooked_layer: Hook at the Layer from the Neural Network Module
        layer_type: Type of layer that will get the MC samples. Either FC (Fully Connected) or
            Conv (Convolutional)
        device: CUDA or CPU device
        reduction_method: Whether to use fullmean, mean, or avgpool to reduce dimensionality
            of hooked representation
        return_raw_predictions: Return or not network outputs

    Returns:
        Monte-Carlo Dropout samples for the input dataloader
    """

    def __init__(
            self,
            model,
            mcd_nro_samples: int,
            hooked_layer: Hook,
            layer_type: str,
            device: str,
            reduction_method: str,
            return_raw_predictions: bool = False,
            drop_probs: Union[float, List] = 0.0,
            dropblock_sizes: Union[int, List] = 0,
            hook_layer_output: bool = True
    ):
        assert layer_type in ("FC", "Conv"), "Layer type must be either 'FC' or 'Conv'"
        assert reduction_method in (
            "mean",
            "fullmean",
        ), "Only mean and fullmean reduction methods supported"
        assert isinstance(model, torch.nn.Module), "model must be a pytorch model"
        assert isinstance(mcd_nro_samples, int), "mcd_nro_samples must be an integer"
        assert isinstance(hooked_layer, Hook), "hook_dropout_layer must be an Hook"
        self.model = model
        self.mcd_nro_samples = mcd_nro_samples
        self.hook_dropout_layer = hooked_layer
        self.layer_type = layer_type
        self.device = device
        self.reduction_method = reduction_method
        self.return_raw_predictions = return_raw_predictions
        self.hook_layer_output = hook_layer_output
        try:
            self.dropout_n_layers = len(drop_probs)
        except TypeError:
            self.dropout_n_layers = 1
            drop_probs = [drop_probs]
            dropblock_sizes = [dropblock_sizes]

        if self.layer_type == "Conv":
            self.dropout_layers = torch.nn.ModuleList(
                DropBlock2D(
                    drop_prob=drop_probs[i],
                    block_size=dropblock_sizes[i]
                ) for i in range(self.dropout_n_layers)
            )
        # FC
        else:
            self.dropout_layers = torch.nn.ModuleList(
                Dropout(drop_probs[i]) for i in range(self.dropout_n_layers)
            )

    def get_ls_mcd_samples(
            self, data_loader: Union[DataLoader, Any]
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Perform the Monte Carlo Dropout inference given a dataloader

        Args:
            data_loader: DataLoader or Dataloader like. Already tested with LoadImages class
            from ultralytics

        Returns:
            Latent MCD samples and optionally the raw inference results
        """
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Extracting MCD samples") as pbar:
                dl_imgs_latent_mcd_samples = []
                if self.return_raw_predictions:
                    raw_predictions = []
                for i, (impath, image, label, im_counter) in enumerate(data_loader):
                    if self.return_raw_predictions:
                        latent_samples, raw_pred = self._get_mcd_samples_one_image(
                            image=image[0]
                        )
                        dl_imgs_latent_mcd_samples.append(latent_samples)
                        raw_predictions.append(raw_pred)
                    else:
                        dl_imgs_latent_mcd_samples.append(
                            self._get_mcd_samples_one_image(image=image[0])
                        )
                    # Update progress bar
                    pbar.update(1)
            dl_imgs_latent_mcd_samples_t = torch.cat(dl_imgs_latent_mcd_samples, dim=0)
        print("MCD N_samples: ", dl_imgs_latent_mcd_samples_t.shape[1])
        if self.return_raw_predictions:
            return dl_imgs_latent_mcd_samples_t, torch.cat(raw_predictions, dim=0)
        else:
            return dl_imgs_latent_mcd_samples_t

    def _get_mcd_samples_one_image(self, image):
        img_mcd_samples = []
        # Perform inference just once per image
        # pred_img = self.model(image)
        pred_img = getattr(self.model, "predict")(source=image, verbose=False)
        if self.hook_layer_output:
            latent_mcd_sample = self.hook_dropout_layer.output
        else:
            latent_mcd_sample = self.hook_dropout_layer.input
            # Input might be a one-element tuple, containing the desired list
            if len(latent_mcd_sample) == 1 and self.dropout_n_layers != 1:
                try:
                    assert len(latent_mcd_sample[0]) == self.dropout_n_layers
                    latent_mcd_sample = latent_mcd_sample[0]
                except AssertionError:
                    print("Cannot find a suitable latent space sample")
        for s in range(self.mcd_nro_samples):
            # Apply dropout/dropblock first
            if self.dropout_n_layers == 1:
                latent_mcd_sample_reduced = self.dropout_layers(latent_mcd_sample)
            else:
                latent_mcd_sample_reduced = [self.dropout_layers[i](latent_mcd_sample[i]) for i
                                             in range(self.dropout_n_layers)]
            if self.dropout_n_layers == 1:
                if self.layer_type == "Conv":
                    latent_mcd_sample_reduced = get_mean_or_fullmean_ls_sample(
                        latent_mcd_sample_reduced,
                        self.reduction_method
                    )
                # FC
                else:
                    # It is already a 1d tensor
                    latent_mcd_sample_reduced = torch.squeeze(latent_mcd_sample_reduced)
            else:
                if self.layer_type == "Conv":
                    n_latent_layers = []
                    for i in range(self.dropout_n_layers):
                        n_latent_layers.append(
                            get_mean_or_fullmean_ls_sample(
                                latent_mcd_sample_reduced[i],
                                self.reduction_method
                            )
                        )
                    latent_mcd_sample_reduced = torch.cat(n_latent_layers, dim=1)
                # FC
                else:
                    raise NotImplementedError
            img_mcd_samples.append(latent_mcd_sample_reduced)

        if self.layer_type == "Conv":
            img_mcd_samples_t = torch.cat(img_mcd_samples, dim=0)
        else:
            img_mcd_samples_t = torch.stack(img_mcd_samples, dim=0)
        if self.return_raw_predictions:
            return img_mcd_samples_t, pred_img
        else:
            return img_mcd_samples_t
