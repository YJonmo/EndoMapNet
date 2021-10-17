
# EndoMapNet
This is the reference implementation for training and testing depth and pose estimation models using the method described in

> **3D Semantic Mapping from Arthroscopy Using Out-of-Distribution Pose and Depth and In-Distribution Segmentation Training**
Jonmohamadi et al, MICCAI 2021.
> [Yaqub Jonmohamadi et al]
>
> [MICCAI 2021]

This repository is modified version of [Monodepth2](https://github.com/nianticlabs/monodepth2) in order to incorportate pose supervision in the original self supervised implementation. Please reference the mentioned papers if you you are using this repository. 

For sample use, please refer to the colab link:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13wVWBbMGv6unxN-NunVLSbl9ywwTdY8X?usp=sharing)


## ⚙️ Setup

conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4

It is tested on Ubuntu 18 and cuda 10. 

TO DO LIST:

More Google colab training examples

