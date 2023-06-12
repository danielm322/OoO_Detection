# CEA-LSEA: Out-of-Distribution & Anomaly Detection using DNN Latent Representation Uncertainty

## Description
CEA-LSEA example for the ***Out-of-Distribution (OoD) detection in DNN latent space*** package.

The current example covers the following DNN architectures for semantic segmentation:

    - Deeplabv3+
    
The `Deeplabv3+` DNN was slightly modified to capture _epistemic_ uncertainty using the Monte-Carlo Dropout approach, and adding a ``DropBlock2D`` layer.


## Requirements DNN
Overview of required deep learning libraries to train/eval the DNN model:

```txt
albumentations==1.3.0
dropblock==0.3.0
pytorch-lightning==1.7.2
lightning-bolts==0.5.0
rich==12.5.1
tensorboard==2.10.0
torch==1.12.1
torchmetrics==0.10.0rc0
torchvision==0.13.1
tqdm==4.64.0
```

For model training. fine-tuning or evaluation, use the `train_deeplab_v3p.py` script. For example:

```bash
>$ python3 train_deeplab_v3p.py -m deeplabv3p-backbone-dropblock2d -b 2 -e 30 --loss_type focal_loss -d woodscape -p /your_path_to_dataset/WoodScape
```

## Requirements OoD/Anomaly detection
Install and use our **CEA OoD & Anomaly detection library**:


```bash
>$ git clone https://repository-url/dnn_ood_detection.git
>$ mv dnn_ood_detection
>$ pip install .

```
To train the OoD detector, we assume access to In-Distribution (InD) and OoD Samples. To generate a dataset with synthetic anomalies, check the datasets scripts and look for the anomaly transformations.

## Usage OoD anomaly detection library
For a detailed usage, check this [document](./Guide%20CEA-LSEA%20OoD%20%26%20Anomaly%20Detection%20DNN%20Latent%20Representaitons.md).

In addition, two jupyter notebooks are available showing the general steps and obtained results in detail:

```txt
- NB_1_ood_anomaly_semseg_deeplabv3p_ws_samples.ipynb
- NB_2_ood_anomaly_semseg_deeplabv3p_ws_ds_shift_detect.ipynb
```

## Publications and Technical Reports
For more technical and implementation details, we refer the user to the following technical
reports and publications:

Technical Reports:
- EC3-FA06 Run-Time Monitoring
- EC3-FA18 Run-Time Monitoring

Publications
- [Out-of-Distribution Detection using Deep Neural Network Latent Space](https://ceur-ws.org/Vol-3381/39.pdf)

## Citing this work

```txt
@article{arnez2023out,
  title={Out-of-Distribution Detection Using Deep Neural Network Latent Space Uncertainty},
  author={Arnez, Fabio and Radermacher, Ansgar and Terrier, Fran{\c{c}}ois},
  year={2023}
}
```

## License
SPDX-License-Identifier: MIT

See also LICENSE.txt
