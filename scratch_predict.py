# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from evaluate_pose_custom2 import Evaluate
from options import MonodepthOptions

import os
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

class Selff(object):
    pass


options = MonodepthOptions()
opts = options.parse()

opts.data_path="/home/jonmoham/DataForTraining/3DPrintedKnee/Air/Raw_Rectif/"
opts.load_weights_folder="/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono01_posecnnAll/mdp/models/weights_19"
opts.log_dir="/home/jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono01_posecnnAll/"
opts.split="2020-05-26-16-07-19"
opts.dataset="Custom"
opts.png=True
opts.height=256
opts.width=256
#opts.models_to_load=['encoder', 'depth', 'pose', 'pose_encoder']
opts.models_to_load=['encoder', 'depth', 'pose']
opts.pose_model_input="all"
#opts.frame_ids=[0, -1, -2, -3, 1, 2, 3]
opts.frame_ids=[0, 1]
opts.use_stereo=True
opts.batch_size=1
#opts.pose_model_type="separate_resnet"
opts.pose_model_type="posecnn"
opts.num_workers = 1

selff = Selff() 


selff.opt = opts
selff.log_path = os.path.join(selff.opt.log_dir, selff.opt.model_name)

# checking height and width are multiples of 32
assert selff.opt.height % 32 == 0, "'height' must be a multiple of 32"
assert selff.opt.width % 32 == 0, "'width' must be a multiple of 32"

selff.models = {}

selff.device = torch.device("cpu" if selff.opt.no_cuda else "cuda")
print(selff.device)
selff.num_input_frames = len(selff.opt.frame_ids)

selff.opt.frame_ids_sorted = []
for i in selff.opt.frame_ids: 
    selff.opt.frame_ids_sorted.append(i)
selff.opt.frame_ids_sorted.sort()

selff.num_pose_frames = 2 if selff.opt.pose_model_input == "pairs" else selff.num_input_frames
assert selff.opt.frame_ids[0] == 0, "frame_ids must start with 0"

selff.use_pose_net = not (selff.opt.use_stereo and selff.opt.frame_ids == [0])

if selff.opt.use_stereo:
    selff.opt.frame_ids.append("s") # this gives the id 's' to  the other image and it means it is a stereo pair

selff.models["encoder"] = networks.ResnetEncoder(
    selff.opt.num_layers, selff.opt.weights_init == "pretrained")
selff.models["encoder"].to(selff.device)
#selff.parameters_to_train += list(selff.models["encoder"].parameters())

selff.models["depth"] = networks.DepthDecoder(
    selff.models["encoder"].num_ch_enc, selff.opt.scales)
selff.models["depth"].to(selff.device)

if selff.use_pose_net:
    if selff.opt.pose_model_type == "separate_resnet":
        selff.models["pose_encoder"] = networks.ResnetEncoder(
            selff.opt.num_layers,
            False,
            num_input_images=selff.num_pose_frames)
        print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL',selff.num_pose_frames)
        print('selff.models["pose_encoder"].num_ch_enc',selff.models["pose_encoder"].num_ch_enc)

        selff.models["pose_encoder"].to(selff.device)

        #selff.models["pose"] = networks.PoseDecoder(
        #    selff.models["pose_encoder"].num_ch_enc,
        #    num_input_features=1,
        #    num_frames_to_predict_for=2)
        selff.models["pose"] = networks.PoseDecoder(
            selff.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=selff.num_pose_frames)

    elif selff.opt.pose_model_type == "shared":
        selff.models["pose"] = networks.PoseDecoder(
            selff.models["encoder"].num_ch_enc, selff.num_pose_frames)

    elif selff.opt.pose_model_type == "posecnn":
        selff.models["pose"] = networks.PoseCNN(
            selff.num_input_frames if selff.opt.pose_model_input == "all" else 2)

    selff.models["pose"].to(selff.device)

if selff.opt.predictive_mask:
    # Our implementation of the predictive masking baseline has the the same architecture
    # as our depth decoder. We predict a separate mask for each source frame.
    selff.models["predictive_mask"] = networks.DepthDecoder(
        selff.models["encoder"].num_ch_enc, selff.opt.scales,
        num_output_channels=(len(selff.opt.frame_ids) - 1))
    selff.models["predictive_mask"].to(selff.device)


if selff.opt.load_weights_folder is not None:
    #%%selff.load_model()
    """Load model(s) from disk
    """
    selff.opt.load_weights_folder = os.path.expanduser(selff.opt.load_weights_folder)

    assert os.path.isdir(selff.opt.load_weights_folder), \
        "Cannot find folder {}".format(selff.opt.load_weights_folder)
    print("loading model from folder {}".format(selff.opt.load_weights_folder))

    for n in selff.opt.models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(selff.opt.load_weights_folder, "{}.pth".format(n))
        model_dict = selff.models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        selff.models[n].load_state_dict(model_dict)

    # loading adam state
    optimizer_load_path = os.path.join(selff.opt.load_weights_folder, "adam.pth")
    if os.path.isfile(optimizer_load_path):
        print("Loading Adam weights")
        optimizer_dict = torch.load(optimizer_load_path)
        #selff.model_optimizer.load_state_dict(optimizer_dict)
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")
    
    
    
selff.models["pose"].to(selff.device)
    
    

print("Training model named:\n  ", selff.opt.model_name)
print("Models and tensorboard events files are saved to:\n  ", selff.opt.log_dir)
print("Training is using:\n  ", selff.device)

# data
datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                 "kitti_odom": datasets.KITTIOdomDataset,
                 "Custom": datasets.Custom}
selff.dataset = datasets_dict[selff.opt.dataset]



print('selff.opt.split: ' + selff.opt.split)
#fpath = os.path.join(os.path.dirname(__file__), "splits", selff.opt.split, "_files.txt")
fpath = os.path.join("/home/jonmoham/Python/monodepth2-master", "splits", selff.opt.split, selff.opt.split+".txt")

print('fpath fpath fpath: ' + fpath)
#print('os.path.dirname(__file__): ' + "/home/jonmoham/Python/monodepth2-master")


train_filenames = readlines(fpath.format("train"))
#print(train_filenames)
#val_filenames = readlines(fpath.format("val"))
img_ext = '.png' if selff.opt.png else '.jpg'

num_train_samples = len(train_filenames)
selff.num_total_steps = num_train_samples // selff.opt.batch_size * selff.opt.num_epochs

train_dataset = selff.dataset(selff.opt.data_path, train_filenames, selff.opt.height, selff.opt.width,
    selff.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
selff.train_loader = DataLoader(train_dataset, selff.opt.batch_size, shuffle=False,
    num_workers=selff.opt.num_workers, pin_memory=True, drop_last=False)

#%%
"""Run the entire evaluation pipeline
"""
selff.epoch = 0
selff.step = 0
#%%selff.run_epoch()

#%%selff.set_eval()
for m in selff.models.values():
    m.eval()

selff.pred_poses = []
batch_idx = 0 
with torch.no_grad():
    for inputs in selff.train_loader:
    #for batch_idx, inputs in enumerate(selff.train_loader):
        print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx))

        batch_idx +=1
#        if batch_idx==1:
#            break
    
        #%%outputs = selff.process_batch(inputs)
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(selff.device)
            #print(key)
        if selff.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in selff.opt.frame_ids])
            all_features = selff.models["encoder"](all_color_aug)
            all_features = [torch.split(f, selff.opt.batch_size) for f in all_features]
    
            features = {}
            for i, k in enumerate(selff.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
    
            outputs = selff.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = selff.models["encoder"](inputs["color_aug", 0, 0])
            outputs = selff.models["depth"](features)
    
    
        if selff.opt.predictive_mask:
            outputs["predictive_mask"] = selff.models["predictive_mask"](features)
    
        if selff.use_pose_net:
            #%%outputs.update(selff.predict_poses(inputs, features))
            """Predict poses between input frames for monocular sequences.
            """
            pred_poses = []
            #outputs = {}
            if selff.num_pose_frames == 2:
                # In this setting, we compute the pose to each source frame via a
                # separate forward pass through the pose network.
    
                # select what features the pose network takes as input
                if selff.opt.pose_model_type == "shared": # if shared is used then the features are the same as features are obtained from the depth encoder
                    pose_feats = {f_i: features[f_i] for f_i in selff.opt.frame_ids}
                else:# if shared is not used then instead of feature, images are given to the pose decoder
                    pose_feats         = {f_i: inputs[ "color_aug", f_i, 0] for f_i in selff.opt.frame_ids}
    
                for f_i in selff.opt.frame_ids[1:]:
                    if f_i != "s":
                        # To maintain ordering we always pass frames in temporal order
                        if f_i < 0:
                            pose_inputs = [pose_feats[f_i], pose_feats[0]]
                        else:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]
    
                        if selff.opt.pose_model_type == "separate_resnet":
                            pose_inputs = [selff.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        elif selff.opt.pose_model_type == "posecnn":
                            pose_inputs = torch.cat(pose_inputs, 1)
    
                        axisangle, translation = selff.models["pose"](pose_inputs)
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
    
                        # Invert the matrix if the frame id is negative
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    
    
            else:
                # Here we input all frames to the pose net (and predict all poses) together
                if selff.opt.pose_model_type in ["separate_resnet", "posecnn"]:
#                    pose_inputs= torch.empty(torch.squeeze(inputs[("color_aug", 0, 0)]).size()[0]
#                        * selff.opt.batch_size * len(selff.opt.frame_ids_sorted), 
#                        torch.squeeze(inputs[("color_aug", 0, 0)]).size()[1], 
#                        torch.squeeze(inputs[("color_aug", 0, 0)]).size()[2], dtype=torch.float)
#                    pose_inputs.cuda() 

#                    for f_i in selff.opt.frame_ids_sorted[:]:
#                        pose_inputs.append(inputs[("color_aug", f_i, 0)])
                        
                        
                    pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in selff.opt.frame_ids_sorted if i != "s"], 1)
    
                    if selff.opt.pose_model_type == "separate_resnet":
#                        for Index, I2  in enumerate(selff.opt.frame_ids):
#                            if I2 != 's':
#                                print(I2)
#                                pose_inputs.append(pose_feats[I2])
                            
                        #pose_inputs = [pose_feats[-1], pose_feats[0], pose_feats[1]]
                        #pose_inputs = [selff.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        pose_inputs = [selff.models["pose_encoder"](pose_inputs)]
    
                elif selff.opt.pose_model_type == "shared":
                    pose_inputs = [features[i] for i in selff.opt.frame_ids if i != "s"]
    
                axisangle, translation = selff.models["pose"](pose_inputs)
                #print(selff.opt.pose_model_type)
                #print(axisangle.size())
                #print(translation.size())
                #print(pose_inputs)
                for i, f_i in enumerate(selff.opt.frame_ids[1:]):
                    if f_i != "s":
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, i], translation[:, i])
       
        
        print('outputs[("cam_T_cam", 0, f_i)][:]')
        print(outputs[("cam_T_cam", 0, 1)][:])
        for i, f_i in enumerate(selff.opt.frame_ids[1:]):
            if f_i != "s":
                selff.pred_poses.append((outputs[("cam_T_cam", 0, f_i)]).cpu().detach().numpy())
        
    selff.pred_poses = np.concatenate(selff.pred_poses)
    save_path = os.path.join(selff.opt.load_weights_folder, "poses_" + selff.opt.split[:]+".npy")
    np.save(save_path, selff.pred_poses)
    print("-> Predictions saved to", save_path)    





