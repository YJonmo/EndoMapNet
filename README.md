
# EndoMapNet
This is the reference implementation for training and testing depth and pose estimation models using the method described in

> **3D Semantic Mapping from Arthroscopy Using Out-of-Distribution Pose and Depth and In-Distribution Segmentation Training**
Jonmohamadi et al, MICCAI 2021.
> [Yaqub Jonmohamadi et al]
>
> [MICCAI 2021]

This repository is modified version of [Monodepth2](https://github.com/nianticlabs/monodepth2) in order to incorportate pose supervision in the original self supervised implementation. Please reference the mentioned papers if you you are using this repository. 

## ⚙️ Setup

conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4

It is tested on Ubuntu 18 and cuda 10. 

A pretrained model on limited number of images (~8300) is provided. For your application further training might be required. 
Sample training and validation data are provided. The training options for this model are:
--dataset Custom --split 3DPrint --png --height 256 --width 256 --frame_ids 0 2 --disparity_smoothness .01  
--use_pose 1 --use_stereo --pose_model_input all


## Prediction for a pose and images

Following commands show how to get the disparity and poses for sample images proveded:

```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

or, if you are using a stereo-trained model, you can estimate metric depth with

```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192 --pred_metric_depth
```

TO DO LIST:

Sample training use

Google colab training examples

Pretrained model
