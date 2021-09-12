#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:42:12 2020

@author: yaqub
"""


from __future__ import absolute_import, division, print_function
from options import MonodepthOptions
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

opts.data_path="/home/jonmoham/DataForTraining/"
opts.log_dir="ScratchTraining"
#opts.split="3Dprint_waterPose_Mono_sub"
opts.split="2020-09-18-13-12-14"
opts.dataset="Custom"
opts.png=True
opts.height=256
opts.width=256
opts.pose_model_input="all"
opts.num_epochs=150
opts.frame_ids=[0,  -5, -3, -1, 1,4]
opts.batch_size=4
opts.learning_rate=0.0005
opts.pose_model_type="separate_resnet"
#opts.pose_model_type="posecnn"
opts.use_stereo=True
opts.use_pose='1'
opts.trans_weight=torch.from_numpy(np.array([.5, 1, .7]))


#opts.load_weights_folder='/home/jonmoham/Python/monodepth2-master/3Dprint_water_SheepPose_Sub0-7-5-3-11357_pairs_Bch14_Lr2-4_2Loss_all/mdp/models/weights_46'
#opts.models_to_load=["pose_encoder2", "pose2"]


selff = Selff() 
selff.opt = opts
selff.log_path = os.path.join(selff.opt.log_dir, selff.opt.model_name)

# checking height and width are multiples of 32
assert selff.opt.height % 32 == 0, "'height' must be a multiple of 32"
assert selff.opt.width % 32 == 0, "'width' must be a multiple of 32"

selff.models = {}
selff.parameters_to_train = []

selff.device = torch.device("cpu" if selff.opt.no_cuda else "cuda")
print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
print(selff.device)
selff.num_scales = len(selff.opt.scales)
selff.opt.frame_ids_sorted = []
for i in selff.opt.frame_ids: 
    selff.opt.frame_ids_sorted.append(i)
selff.opt.frame_ids_sorted.sort()
selff.num_input_frames = len(selff.opt.frame_ids)
selff.num_pose_frames = 2 if selff.opt.pose_model_input == "pairs" else selff.num_input_frames

if selff.opt.pose_model_input == "pairs":
    selff.num_frames_to_predict_for = 2
elif (selff.opt.pose_model_input == "all") and (selff.opt.use_stereo):
    selff.num_frames_to_predict_for = selff.num_input_frames - 1 

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%selff.num_pose_frames', selff.num_pose_frames)
assert selff.opt.frame_ids[0] == 0, "frame_ids must start with 0"

selff.use_pose_net = not (selff.opt.use_stereo and selff.opt.frame_ids == [0])

if selff.opt.use_stereo:
    selff.opt.frame_ids.append("s") # this gives the id 's' to  the other image and it means it is a stereo pair
#
#selff.models["encoder"] = networks.ResnetEncoder(
#    selff.opt.num_layers, selff.opt.weights_init == "pretrained")
#selff.models["encoder"].to(selff.device)
#selff.parameters_to_train += list(selff.models["encoder"].parameters())
#
#selff.models["depth"] = networks.DepthDecoder(
#    selff.models["encoder"].num_ch_enc, selff.opt.scales)
#selff.models["depth"].to(selff.device)
#selff.parameters_to_train += list(selff.models["depth"].parameters())

#selff.criterion = torch.nn.MSELoss()
selff.criterion = torch.nn.L1Loss()
#selff.criterion = torch.nn.CrossEntropyLoss()
#selff.criterion = torch.nn.PoissonNLLLoss()

selff.parameters_to_train2 = []

#if selff.opt.pose_model_type == "separate_resnet":
#    selff.models["pose_encoder2"] = networks.ResnetEncoder(
#        selff.opt.num_layers,
#        selff.opt.weights_init == "pretrained",
#        num_input_images=selff.num_pose_frames)
if selff.opt.pose_model_type == "separate_resnet":
    selff.models["pose_encoder2"] = networks.ResnetEncoder(
        selff.opt.num_layers,
        selff.opt.weights_init == "pretrained",
        num_input_images=selff.num_pose_frames)
    selff.models["pose_encoder2"].to(selff.device)
    selff.parameters_to_train2 += list(selff.models["pose_encoder2"].parameters())

    selff.models["pose2"] = networks.PoseDecoder(
        selff.models["pose_encoder2"].num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=selff.num_frames_to_predict_for)

elif selff.opt.pose_model_type == "posecnn":
    selff.models["pose2"] = networks.PoseCNN(
        selff.num_input_frames if selff.opt.pose_model_input == "all" else 2)



selff.models["pose2"].to(selff.device)
selff.parameters_to_train2 += list(selff.models["pose2"].parameters())
selff.model_optimizer2 = optim.Adam(selff.parameters_to_train2, selff.opt.learning_rate)
#selff.model_optimizer2 = optim.SGD(selff.parameters_to_train2, selff.opt.learning_rate, momentum=0.9, weight_decay=0.0001)
selff.model_lr_scheduler2 = optim.lr_scheduler.StepLR(
    selff.model_optimizer2, selff.opt.scheduler_step_size, 0.1)

if selff.opt.load_weights_folder is not None:
    #selff.load_model()
    
    #def load_model(self):
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
        selff.model_optimizer2.load_state_dict(optimizer_dict)
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")
    
    
    
    

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
fpath = os.path.join("/home/jonmoham/Python/monodepth2-master", "splits", selff.opt.split, "{}_files.txt")
print('fpath fpath fpath: ' + fpath)
#print('os.path.dirname(__file__): ' + "/home/jonmoham/Python/monodepth2-master")

train_filenames = readlines(fpath.format("train"))
#val_filenames = readlines(fpath.format("val"))
img_ext = '.png' if selff.opt.png else '.jpg'

num_train_samples = len(train_filenames)
selff.num_total_steps = num_train_samples // selff.opt.batch_size * selff.opt.num_epochs

#train_dataset = selff.dataset(
#    selff.opt.data_path, train_filenames, selff.opt.height, selff.opt.width,
#    selff.opt.frame_ids, 4, is_train=True, img_ext=img_ext)

train_dataset = selff.dataset(
    selff.opt.data_path, train_filenames, selff.opt.height, selff.opt.width,
    selff.opt.frame_ids, 4, is_train=True, img_ext=img_ext, use_pose=selff.opt.use_pose)

selff.train_loader = DataLoader(
    train_dataset, selff.opt.batch_size, True,
    num_workers=selff.opt.num_workers, pin_memory=True, drop_last=True)
    #val_dataset = selff.dataset(
    #    selff.opt.data_path, val_filenames, selff.opt.height, selff.opt.width,
    #    selff.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
    #selff.val_loader = DataLoader(
    #    val_dataset, selff.opt.batch_size, True,
    #    num_workers=selff.opt.num_workers, pin_memory=True, drop_last=True)
    #selff.val_iter = iter(selff.val_loader)

selff.writers = {}
for mode in ["train", "val"]:
    selff.writers[mode] = SummaryWriter(os.path.join(selff.log_path, mode))

if not selff.opt.no_ssim:
    selff.ssim = SSIM()
    selff.ssim.to(selff.device)

selff.backproject_depth = {}
selff.project_3d = {}
for scale in selff.opt.scales:
    h = selff.opt.height // (2 ** scale)
    w = selff.opt.width // (2 ** scale)

    selff.backproject_depth[scale] = BackprojectDepth(selff.opt.batch_size, h, w)
    selff.backproject_depth[scale].to(selff.device)

    selff.project_3d[scale] = Project3D(selff.opt.batch_size, h, w)
    selff.project_3d[scale].to(selff.device)

selff.depth_metric_names = [
    "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

print("Using split:\n  ", selff.opt.split)
#print("There are {:d} training items and {:d} validation items\n".format(
#    len(train_dataset), len(val_dataset)))
print("There are {:d} training items\n".format(
    len(train_dataset)))

#selff.save_opts()    




#%%



"""Run the entire training pipeline
"""
selff.epoch = 0
selff.step = 0
selff.start_time = time.time()
#%%####################selff.run_epoch()




"""Run a single epoch of training and validation
"""
selff.model_lr_scheduler2.step()

print("Training")
#%%selff.set_train()


"""Convert all models to training mode
"""
for m in selff.models.values():
    m.train()


selff.opt.trans_weight = selff.opt.trans_weight.to(selff.device)
#with torch.autograd.set_detect_anomaly(True):

for selff.epoch in range(selff.opt.num_epochs):
    #self.run_epoch()        
    for batch_idx, inputs in enumerate(selff.train_loader):
        
        print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx) + str('selff.epoch')+ str(selff.epoch))
        #print(inputs)
#        if batch_idx == 0:
#            break
    
        #%%outputs, losses = selff.process_batch(inputs)
        
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(selff.device)
            #print(key)
    
    
    
        if selff.use_pose_net:
            #%%outputs.update(selff.predict_poses(inputs, features))
            """Predict poses between input frames for monocular sequences.
            """
            pred_poses = []
            outputs2 = {}
            if selff.num_pose_frames == 2:
                # In this setting, we compute the pose to each source frame via a
                # separate forward pass through the pose network.
        
    
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in selff.opt.frame_ids}
                #print('###########################################Pose_Feature###############################')
                #print(pose_feats)
                for f_i in selff.opt.frame_ids[1:]:
                    if f_i != "s":
                        # To maintain ordering we always pass frames in temporal order
                        if f_i < 0:
                            pose_inputs2 = [pose_feats[f_i], pose_feats[0]]
                        else:
                            pose_inputs2 = [pose_feats[0], pose_feats[f_i]]
    



                        if selff.opt.pose_model_type == "separate_resnet":
                            pose_inputs2 = [selff.models["pose_encoder2"](torch.cat(pose_inputs2, 1))]

                        elif selff.opt.pose_model_type == "posecnn":
                            pose_inputs2 = torch.cat(pose_inputs2, 1)
                        selff.model_optimizer2.zero_grad()   # clear the buffer
                        
                        axisangle2, translation2 = selff.models["pose2"](pose_inputs2)
                        if (f_i < 0):
                            translation2 = translation2*(-1)
                        outputs2[("axisangle", 0, f_i)] = axisangle2[:, 0, 0 , :]
                        outputs2[("translation", 0, f_i)] = translation2[:, 0, 0 , :]

                        # Invert the matrix if the frame id is negative
                        #outputs2[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        #    axisangle2[:, 0], translation2[:, 0], invert=(f_i < 0))    

            else:
                # Here we input all frames to the pose net (and predict all poses) together
                if selff.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                    pose_inputs2 = torch.cat([inputs[("color_aug", i, 0)] for i in selff.opt.frame_ids_sorted if i != "s"], 1)
                    
                    if selff.opt.pose_model_type == "separate_resnet":
                        pose_inputs2 = [selff.models["pose_encoder2"](pose_inputs2)]

    
                axisangle2, translation2 = selff.models["pose2"](pose_inputs2)
                for i, f_i in enumerate(selff.opt.frame_ids[1:]):
                    if f_i != "s":
                        outputs2[("axisangle", 0, f_i)] = axisangle2[:, i, 0 , :]
                        outputs2[("translation", 0, f_i)] = translation2[:, i, 0 , :]
                        outputs2[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle2[:, i, : , :], translation2[:, i, : , :])
            #return outputs





     

        frame_lengh = 0
        for frame_id in (selff.opt.frame_ids[1:]):
            if frame_id != "s":
                frame_lengh +=1   
        translation_gt = torch.zeros(selff.opt.batch_size, 3, frame_lengh, requires_grad=True, dtype=torch.float64) 
        translation_gt = translation_gt.to(device='cuda')
        translation_pred = torch.zeros(selff.opt.batch_size, 3, frame_lengh, requires_grad=False, dtype=torch.float64) 
        translation_pred = translation_pred.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float64) 
        i=0
        for frame_id in (selff.opt.frame_ids[1:]):
            if frame_id != "s":
                #axisangle_pred = outputs[("axisangle", 0, frame_id)]
                translation_pred[:, :, i] = outputs2[("translation", 0, frame_id)][:,:]*100
                #axisangle_gt = inputs[("axisangle", frame_id, 0)]
                #Sign = torch.sign(inputs[("translation", frame_id, 0)])
                #translation_gt = (-1*Sign)*torch.log10(torch.abs(inputs[("translation", frame_id, 0)]))/100
                translation_gt[:, :, i] = inputs[("translation", frame_id, 0)]*(1/1)
                print('\n translation_pred[0,:,i].abs().mean(): ')
                print(translation_pred[0,:,i].data) 
                print('translation_gt[0,:,i].abs().mean(): ')
                print(translation_gt[0,:,i].data)    
                #abs_diff_trans[i] = (torch.squeeze(translation_pred[:,0]) - translation_gt + 0.000000).abs().mean()
                #abs_diff_trans[:, :, i] = (torch.squeeze(translation_pred) - translation_gt + 0.00)
                #abs_diff_axang.append(torch.abs(axisangle_pred[:,0]   - axisangle_gt).abs().mean())
                #abs_diff_trans[i] = (translation_pred[:,0,0,:] - translation_gt[:,:]).square().sum().sqrt()
                i += 1

        for Bach_ind in range(translation_gt.size()[0]):
            for Frame_ind in range(translation_gt.size()[2]):
                translation_gt[Bach_ind, :, Frame_ind] = translation_gt[Bach_ind, :, Frame_ind]*selff.opt.trans_weight
        
        losses = {}              
        losses["loss2_Normal"]=selff.criterion(F.normalize(translation_gt), F.normalize(translation_pred))       
        losses["loss2_Raw"] = selff.criterion(translation_gt, translation_pred)   
        print("loss2_Normal: " + str(losses["loss2_Normal"].data))   
        print("loss2_Raw: "+ str(losses["loss2_Raw"].data))         
        #coeff = torch.zeros(1, requires_grad=True, dtype=torch.float64)  
        #coeff = coeff.to(device='cuda')     
        #coeff[0] = 1
        #losses = {}
        #losses["loss2"] = abs_diff_trans.mean() * coeff
        #losses["loss2"] = abs_diff_trans.abs().mean() * coeff
        #losses["loss2"] = selff.criterion(translation_pred[:],translation_gt[:])/frame_lengh

        #losses2 = outputs[("translation", 0, f_i)][:,0].mean()
        losses["loss2"] = (losses["loss2_Normal"] + losses["loss2_Raw"])/frame_lengh
        #selff.model_optimizer2.zero_grad()   # clear the buffer
        if not(torch.isnan(losses["loss2"])):
            losses["loss2"].backward()           # back propagate the loss 
            
            selff.model_optimizer2.step()        # update the weights
        
        #print('losses[loss2]: ' + str(losses["loss2"]*100))
    
        
        #selff.log("train", inputs, outputs, losses)
        #def log(self, mode, inputs, outputs, losses):
        if np.remainder(selff.step+5,40) == 0:
            """Write an event to the tensorboard events file
            """
            writer = selff.writers["train"]
            for l, v in losses.items():
                writer.add_scalar("{}".format(l), v, selff.step)
    
            for j in range(min(4, selff.opt.batch_size)):  # write a maxmimum of four images
                for s in selff.opt.scales:
                    for frame_id in selff.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, selff.step)

        selff.step += 1

    #%%####################################################################################################################     
    
    #if (self.epoch + 1) % self.opt.save_frequency == 0:
    #self.save_model()
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    
#    
#        # log less frequently after the first 2000 steps to save time & disk space
#        early_phase = batch_idx % selff.opt.log_frequency == 0 and selff.step < 2000
#        late_phase = selff.step % 2000 == 0
#    
#        if early_phase or late_phase:
#            selff.log_time(batch_idx, duration, losses["loss"].cpu().data)
#    
#            if "depth_gt" in inputs:
#                selff.compute_depth_losses(inputs, outputs, losses)
#    
#            selff.log("train", inputs, outputs, losses)
#            #selff.val()
#    



