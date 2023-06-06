# CEA-LSEA: Out-of-Distribution Detection using DNN Latent Representations Uncertainty

## Description

CEA-LSEA package for Out-of-Distribution (OoD) detection using the uncertainty (entropy) from DNN latent representations.
The package has been used with the following applications, the corresponding DNN architectures and datasets:

- **Simple Classification:**
    - **In-Distribution Dataset:** GTSRB
    - **Out-of-Distribution Datasets:** CIFAR10 & STL10
    - **DNN Architectures:**
        1. ResNet-18
        2. ResNet-18 with Spectral Normalization 

- **Semantic Segmentation:**
    - **In-Distribution Dataset:** Woodscape  & Cityscapes
    - **Out-of-Distribution Datasets:** Woodscape soiling, Woodscape-anomalies, Cityscapes-anomalies
    - **DNN Architectures:**
      1. Deeplabv3+
      2. U-Net
    

In all the above cases, the DNNs were slightly modified to capture _epistemic_ uncertainty using the Monte-Carlo Dropout by adding a ``DropBlock2D`` layer.


## Requirements
Use python 3.8, then install the `requirements.txt`

## Installation
Install requirements, then in the base folder of the repo do `pip install .`

## Usage
For detailed usage, check this [document](./ls_ood_detect_cea/CEA-LSEA-OoD%20Detection%20DNN%20Latent%20Space.md)

## Publications and Technical Reports
For more technical and implementation details, we refer the user to the following technical
reports and publications:

Technical Reports:

    - EC3-FA06 Run-Time Monitoring
    - EC3-FA18 Run-Time Monitoring

Publications
    
    - Out-of-Distribution Detection using Deep Neural Network Latent Space

## License
ToDo!