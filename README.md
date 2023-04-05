# CEA-LSEA: Out-of-Distribution Detection in DNN Latent Space Uncertainty

## Description
CEA-LSEA package for Out-of-Distribution (OoD) detection in DNN latent space.
The package currently supports the following DNN architectures for semantic segmentation:

    - Deeplabv3+
    - Probabilistic U-Net
    

In both cases, the DNN were slightly modified to capture _epistemic_ uncertainty using the Monte-Carlo Dropout.
approach, and adding a ``DropBlock2D`` layer.

To train the OoD detector, we assume access to In-Distribution (InD) and OoD Samples. 


## Requirements
ToDo!

## Installation
ToDo!

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