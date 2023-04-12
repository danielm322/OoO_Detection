from setuptools import setup, find_packages

VERSION = '1.0.0rc-1'
DESCRIPTION = 'CEA-DNN Latent Space Out-of-Distribution Detection'
LONG_DESCRIPTION = 'CEA package for Out-of-Distribution Detection in DNN Latent Space'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="ls_ood_detect_cea",
    version=VERSION,
    author="Fabio Arnez",
    author_email="<farnez@cea.fr>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    python_requires='>=3.7',
    packages=find_packages(),
    # install_requires=['torch',
    #                   'torchvision',
    #                   'pytorch-lightning',
    #                   'torchmetrics'],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['out-of-distribution detection', 'DNNs'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Confiance-AI Partners",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux"
    ]
)
