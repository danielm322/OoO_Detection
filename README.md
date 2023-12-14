<div align="center">
    <img src="assets/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">Out-of-Distribution Detection using DNN
Latent Representations Uncertainty</h1>
</div><div align="center">
    <a href="https://www.python.org/downloads/release/python-380/">
        <img src="https://img.shields.io/badge/Python-3.8-efefef">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</div>
<br>

---
# Guideline CEA-LSEA Out-of-Distribution Detection using DNN Latent Representations Uncertainty
---

CEA-LSEA package for Out-of-Distribution (OoD) detection using the uncertainty (entropy) from DNN latent representations.
The package has been used with the following applications, the corresponding DNN architectures and datasets:

- **Simple Classification:**
    - **In-Distribution Dataset:** CIFAR10
    - **Out-of-Distribution Datasets:** FMNIST, SVHN, Places365, Textures, iSUN, LSUN-C, LSUN-R
    - **DNN Architectures:**
        1. ResNet-18
        2. ResNet-18 with Spectral Normalization

- **Object Detection:**
    - **In-Distribution Dataset:** BDD100k
    - **Out-of-Distribution Datasets:** Pascal VOC, Openimages
    - **DNN Architectures:**
      1. Faster RCNN

- **Semantic Segmentation:**
    - **In-Distribution Dataset:** Woodscape  & Cityscapes
    - **Out-of-Distribution Datasets:** Woodscape soiling, Woodscape-anomalies, Cityscapes-anomalies
    - **DNN Architectures:**
      1. Deeplabv3+
      2. U-Net

In all the above cases, the DNNs were slightly modified to capture _epistemic_ uncertainty using the Monte-Carlo Dropout by adding a ``DropBlock2D`` layer.

- [🔍 Specifications](#specifications)
- [🚀 Quick Start](#quickstart)
- [🎮 Usage](#usage)
- [🔀 Description of Inputs and Outputs](#io)
- [💻 Required Hardware](#hardware)
- [📚 References](#references)

<div id='specifications'/>

## 🔍 Specifications
- Version: 1.0.0
- Python Version: python 3.8
- Strong Dependencies: torch, entropy_estimators, numpy, sklearn, dropblock, pandas, mlflow, matplotlib
- Thematic: Computer vision
- Trustworthy: Uncertainty Estimation -  OoD/Anomaly detection
- Hardware : GPU


<div id='id-card'/>

## 🔍 Identity card

 Can be found here: [Identity card](./identity_card.yml)

<div id='quickstart'/>

## 🚀 Quick Start

To install and use the component , it is recommended to create a Python virtual environment. You can do that with virtualenv, as follows:

### Setting environement

With `virtualenv`
```bash
pip install virtualenv
virtualenv ls_ood_detection_env
source ls_ood_detection_env/bin/activate
```

Or using conda:
```bash
conda create -n ls_ood_detection_env python=3.8
conda activate ls_ood_detection_env
```


### Installation
After creating the environment with python 3.8, install the `requirements.txt` using:
```bash
pip install -r requirements.txt
```

After installing all the requirements, then in the base folder of the repo do `pip install .`


<div id='usage'/>

## 🎮 Usage

For a complete example of how to use the component, check the notebooks: [Notebook 1: extract MCDz samples](./examples/1_example_deeplab_segmentation_samples_extraction.ipynb),
and [Notebook 2: Evaluate LaREx](./examples/2_example_deeplab_segmentation_analysis.ipynb)

Here we present a general overview of how to evaluate the component and obtain detection metrics:

```python
# Load the trained model
trained_model = MyTrainedModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
# Hook the dropout or dropblock layer
hooked_layer = Hook(trained_model.my_dropblock_layer)
N_MCD_SAMPLES = 16
# Activate dropout or dropblock at inference
trained_model.to(device)
trained_model.eval()
trained_model.apply(deeplabv3p_apply_dropout)

# Extract MCDz samples
# InD
latent_samples_ind_train = get_latent_representation_mcd_samples(trained_model,
                                                                 my_ind_train_data_loader,
                                                                 N_MCD_SAMPLES,
                                                                 hooked_layer)
latent_samples_ind_test = get_latent_representation_mcd_samples(trained_model,
                                                                my_ind_test_data_loader,
                                                                N_MCD_SAMPLES,
                                                                hooked_layer)
# OoD
latent_samples_ood_test = get_latent_representation_mcd_samples(trained_model,
                                                                my_ood_test_data_loader,
                                                                N_MCD_SAMPLES,
                                                                hooked_layer)
# Get entropy
_, entropy_samples_ind_train = get_dl_h_z(latent_samples_ind_train, mcd_samples_nro=N_MCD_SAMPLES)
_, entropy_samples_ind_test = get_dl_h_z(latent_samples_ind_test, mcd_samples_nro=N_MCD_SAMPLES)
_, entropy_samples_ood_test = get_dl_h_z(latent_samples_ood_test, mcd_samples_nro=N_MCD_SAMPLES)

# Evaluate LaRED and LaREM
# Pass OoD samples as dictionary (you can have several OoD datasets)
ood_datasets_dict = {
    'my_ood_dataset1': entropy_samples_ood_test
}
metrics_pandas_df = log_evaluate_lared_larem(
    ind_train_h_z=entropy_samples_ind_train,
    ind_test_h_z=entropy_samples_ind_test,
    ood_h_z_dict=ood_datasets_dict,
)
```

<div id='io'/>

## 🔀 Description of Inputs and Outputs

In general, to perform OoD or anomaly detection with our method you need:
* A well defined In Distribution (InD) dataset
* A model already trained on the InD dataset
* One (or more) dataset(s) defined as Out of Distribution (OoD) or Anomaly.

Then if the model was trained with at least one dropout or dropblock layer, we need to:
* Attach a Hook to the dropblock or dropout layer to catch the outputs of such layer: Use the
  `Hook` class
* Perform Monte Carlo dropout sampling, meaning, inference is performed $N$ times, at each time the
  output of the hooked layer is taken (Use the `get_latent_representation_mcd_samples` function
  (You just need to specify the type of layer in the function parameters):
    * If the layer is convolutional (dropblock layer): we take the mean per channel
    * If the layer is Fully Connected (dropout layer): we take the raw output

* Take the entropy of the previously calculated samples. Use the `get_dl_h_z` function
* Pass the entropy of each image to the LaRED and LaREM estimators to obtain the OoD score. Use the
  `log_evaluate_lared_larem` function.

We will obtain at the end a pandas dataframe with the evaluation metrics and classification
thresholds of LaRED and LaREM. To use the method during inference it is needed to save the density
from LaRED or LaREM,
optionally the PCA transformation, and the threshold chosen for classification. Then we would
obtain the uncertainty estimation at inference time.

<div id='hardware'/>

## 💻 Required Hardware

GPU is a requirement for our component since we deal with "heavy" tasks in computer vision such as
object detection and image segmentation, which are typically too slow in CPU.

<div id='references'/>

## 📚 References

For more technical and implementation details, we refer the user to the following technical
reports and publications:

Technical Reports:

    - EC3-FA06 Run-Time Monitoring
    - EC3-FA18 Run-Time Monitoring

Publications

    - Out-of-Distribution Detection using Deep Neural Network Latent Space


Confiance AI documents:
* [Methodological guidelines](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/ERYc5y-HkPdAvL0TVAQdp0kBkfsPhJwrXrfZrVsH8CuY8Q?e=1mpavP)
* [Benchmarks](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/EfaV2zJlJ9VOqMHSr9sk1JIBvXl3CjQGRzHzwAtO_SXiHQ?e=AbUAiM)
* [Use Case application: Valeo - scene understanding](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/EZKRyjRiobZLm58OoerTgTYB9o_PjyuPpVY7PXFb_v0_hg?e=cWNHdI)