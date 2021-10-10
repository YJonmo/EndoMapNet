# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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
opts.log_dir="3D_Mono0-1-212_Pair_Bach22_LR-5_Smooth.005_Pose/"
opts.split="3Dprint_water_Mono"
opts.dataset="Custom"
opts.png=True
opts.height=256
opts.width=256
opts.pose_model_input="pairs"
opts.num_epochs=1
opts.frame_ids=[0, -1, -2, 1, 2]
opts.batch_size=1

opts.pose_model_type="separate_resnet"

opts.use_pose=True
#opts.load_weights_folder='/home/jonmoham/Python/monodepth2-master/Cad_3D_Stereo_Mono0-1-212_Pairs_Bach22/mdp/models/weights_24'
#opts.models_to_load=["encoder","depth","pose_encoder", "pose"]


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
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%selff.num_pose_frames', selff.num_pose_frames)
assert selff.opt.frame_ids[0] == 0, "frame_ids must start with 0"

selff.use_pose_net = not (selff.opt.use_stereo and selff.opt.frame_ids == [0])

if selff.opt.use_stereo:
    selff.opt.frame_ids.append("s") # this gives the id 's' to  the other image and it means it is a stereo pair

selff.models["encoder"] = networks.ResnetEncoder(
    selff.opt.num_layers, selff.opt.weights_init == "pretrained")
selff.models["encoder"].to(selff.device)
selff.parameters_to_train += list(selff.models["encoder"].parameters())

selff.models["depth"] = networks.DepthDecoder(
    selff.models["encoder"].num_ch_enc, selff.opt.scales)
selff.models["depth"].to(selff.device)
selff.parameters_to_train += list(selff.models["depth"].parameters())

if selff.use_pose_net:
    if selff.opt.pose_model_type == "separate_resnet":
        selff.models["pose_encoder"] = networks.ResnetEncoder(
            selff.opt.num_layers,
            selff.opt.weights_init == "pretrained",
            num_input_images=selff.num_pose_frames)
        print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL',selff.num_pose_frames)
        print('selff.models["pose_encoder"].num_ch_enc',selff.models["pose_encoder"].num_ch_enc)

        selff.models["pose_encoder"].to(selff.device)
        selff.parameters_to_train += list(selff.models["pose_encoder"].parameters())

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
    selff.parameters_to_train += list(selff.models["pose"].parameters())

if selff.opt.predictive_mask:
    # Our implementation of the predictive masking baseline has the the same architecture
    # as our depth decoder. We predict a separate mask for each source frame.
    selff.models["predictive_mask"] = networks.DepthDecoder(
        selff.models["encoder"].num_ch_enc, selff.opt.scales,
        num_output_channels=(len(selff.opt.frame_ids) - 1))
    selff.models["predictive_mask"].to(selff.device)
    selff.parameters_to_train += list(selff.models["predictive_mask"].parameters())

selff.model_optimizer = optim.Adam(selff.parameters_to_train, selff.opt.learning_rate)
selff.model_lr_scheduler = optim.lr_scheduler.StepLR(
    selff.model_optimizer, selff.opt.scheduler_step_size, 0.1)

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
        selff.model_optimizer.load_state_dict(optimizer_dict)
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
selff.model_lr_scheduler.step()

print("Training")
#%%selff.set_train()


"""Convert all models to training mode
"""
for m in selff.models.values():
    m.train()




for batch_idx, inputs in enumerate(selff.train_loader):
    print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx))
    #print(inputs)
    if batch_idx == 0:
        break

    #%%outputs, losses = selff.process_batch(inputs)
    
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
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in selff.opt.frame_ids}
                #print('###########################################Pose_Feature###############################')
                #print(pose_feats)
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
                pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in selff.opt.frame_ids_sorted if i != "s"], 1)
                for key, ipt in inputs.items():
                    #inputs[key] = ipt.to(selff.device)
                    print(key)
    #                for  i in selff.opt.frame_ids: 
    #                    print('iiiiiiiiiiiiiiiiiii is: ' + str(i))
    #                    
    #                    if i != "s":
    #                        pose_inputs = torch.cat([inputs[("color_aug", i, 0)]], 1)
    #                        #pose_inputs = torch.cat(pose_inputs, 1)
    
                if selff.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [selff.models["pose_encoder"](pose_inputs)]
    
            elif selff.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in selff.opt.frame_ids if i != "s"]
    
            axisangle, translation = selff.models["pose"](pose_inputs)
            #print(pose_inputs)
            for i, f_i in enumerate(selff.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])
                    
        print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' + str(selff.opt.frame_ids))        
        print(selff.opt.pose_model_type)
        print(axisangle.size())
        print(translation.size())
        #return outputs    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        #%%selff.generate_images_pred(inputs, outputs)
        for scale in selff.opt.scales:
            disp = outputs[("disp", scale)]
            if selff.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [selff.opt.height, selff.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
    
            _, depth = disp_to_depth(disp, selff.opt.min_depth, selff.opt.max_depth)
    
            outputs[("depth", 0, scale)] = depth
    
            for i, frame_id in enumerate(selff.opt.frame_ids[1:]):
    
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
    
                # from the authors of https://arxiv.org/abs/1712.00175
                if (selff.opt.pose_model_type == "posecnn") and frame_id != "s":
    
                    #if frame_id != "s":
                    #print(outputs[("axisangle", 0, 0)])
                    #print(outputs)
    
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
    
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
    #                    pred_poses = []
    #                    pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().detach().numpy())
    #                    #print('###########################################translation##############################')
    #                    #print(pred_poses)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
    
                cam_points = selff.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = selff.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
    
                outputs[("sample", frame_id, scale)] = pix_coords
    
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
    
                if not selff.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
        
        
      
        #%%losses = selff.compute_losses(inputs, outputs)
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
    
        for scale in selff.opt.scales:
            loss = 0
            reprojection_losses = []
    
            if selff.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
    
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
    
            for frame_id in selff.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                #reprojection_losses.append(selff.compute_reprojection_loss(pred, target))
                """Computes reprojection loss between a batch of predicted and target images
                """
                abs_diff = torch.abs(target - pred)
                l1_loss = abs_diff.mean(1, True)
        
                if selff.opt.no_ssim:
                    reprojection_loss = l1_loss
                else:
                    ssim_loss = selff.ssim(pred, target).mean(1, True)
                    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                    #print('ssim_loss: ' + str(ssim_loss))
                    #print('l1_loss: ' + str(l1_loss))
                reprojection_losses.append(reprojection_loss)
        
            reprojection_losses = torch.cat(reprojection_losses, 1)
    
            if not selff.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in selff.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    #identity_reprojection_losses.append(selff.compute_reprojection_loss(pred, target))
                    
                    """Computes reprojection loss between a batch of predicted and target images
                    """
                    abs_diff = torch.abs(target - pred)
                    l1_loss = abs_diff.mean(1, True)
            
                    if selff.opt.no_ssim:
                        reprojection_loss = l1_loss
                    else:
                        ssim_loss = selff.ssim(pred, target).mean(1, True)
                        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                        #print('ssim_loss: ' + str(ssim_loss))
                        #print('l1_loss: ' + str(l1_loss))
                    identity_reprojection_losses.append(reprojection_loss)
                    
                    
    
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
    
                if selff.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
    
            elif selff.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not selff.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [selff.opt.height, selff.opt.width],
                        mode="bilinear", align_corners=False)
    
                reprojection_losses *= mask
    
                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()
    
            if selff.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
    
            if not selff.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001
    
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
    
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
    
            if not selff.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()
    
            loss += to_optimise.mean()
    
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
    
            loss += selff.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            
            
            
        i = 0
        for frame_id in (selff.opt.frame_ids[1:]):
            if frame_id != "s":
                i +=1   
        abs_diff_trans = torch.zeros(i, dtype=torch.float32) 
        abs_diff_trans = abs_diff_trans.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float32) 
        i=0
        for frame_id in (selff.opt.frame_ids[1:]):
            if frame_id != "s":
                axisangle_pred = outputs[("axisangle", 0, frame_id)]
                translation_pred = outputs[("translation", 0, frame_id)]
                #axisangle_gt = inputs[("axisangle", frame_id, 0)]
                translation_gt = inputs[("translation", frame_id, 0)]
                    
                abs_diff_trans[i] = (translation_pred[:,0] - translation_gt).abs().mean()
                #abs_diff_axang.append(torch.abs(axisangle_pred[:,0]   - axisangle_gt).abs().mean())
                i += 1
                
                
                
        coeff = 0.01        
        l1_loss_pose = abs_diff_trans.mean() * coeff


        total_loss /= selff.num_scales
        
        total_loss += l1_loss_pose
        
        losses["loss"] = total_loss
        #return losses







    selff.model_optimizer.zero_grad()   # clear the buffer
    losses["loss"].backward()           # back propagate the loss 
    selff.model_optimizer.step()        # update the weights


    # log less frequently after the first 2000 steps to save time & disk space
    early_phase = batch_idx % selff.opt.log_frequency == 0 and selff.step < 2000
    late_phase = selff.step % 2000 == 0

    if early_phase or late_phase:
        selff.log_time(batch_idx, duration, losses["loss"].cpu().data)

        if "depth_gt" in inputs:
            selff.compute_depth_losses(inputs, outputs, losses)

        selff.log("train", inputs, outputs, losses)
        #selff.val()
























def set_train(self):
    """Convert all models to training mode
    """
    for m in selff.models.values():
        m.train()

def set_eval(self):
    """Convert all models to testing/evaluation mode
    """
    for m in selff.models.values():
        m.eval()

def train(self):
    

def run_epoch(self):
    """Run a single epoch of training and validation
    """
    selff.model_lr_scheduler.step()

    print("Training")
    selff.set_train()

    for batch_idx, inputs in enumerate(selff.train_loader):
        print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx))
        #print(inputs)
        before_op_time = time.time()

        outputs, losses = selff.process_batch(inputs)

        selff.model_optimizer.zero_grad()
        losses["loss"].backward()
        selff.model_optimizer.step()

        duration = time.time() - before_op_time

        # log less frequently after the first 2000 steps to save time & disk space
        early_phase = batch_idx % selff.opt.log_frequency == 0 and selff.step < 2000
        late_phase = selff.step % 2000 == 0

        if early_phase or late_phase:
            selff.log_time(batch_idx, duration, losses["loss"].cpu().data)

            if "depth_gt" in inputs:
                selff.compute_depth_losses(inputs, outputs, losses)

            selff.log("train", inputs, outputs, losses)
            #selff.val()

        selff.step += 1

def process_batch(self, inputs):
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
        outputs.update(selff.predict_poses(inputs, features))
        print('###########################################Output###############################')
        #print(outputs)
    selff.generate_images_pred(inputs, outputs)
    losses = selff.compute_losses(inputs, outputs)

    return outputs, losses

def predict_poses(self, inputs, features):
    """Predict poses between input frames for monocular sequences.
    """
    pred_poses = []
    outputs = {}
    if selff.num_pose_frames == 2:
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        if selff.opt.pose_model_type == "shared": # if shared is used then the features are the same as features are obtained from the depth encoder
            pose_feats = {f_i: features[f_i] for f_i in selff.opt.frame_ids}
        else:# if shared is not used then instead of feature, images are given to the pose decoder
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in selff.opt.frame_ids}
            #print('###########################################Pose_Feature###############################')
            #print(pose_feats)
        for f_i in selff.opt.frame_ids[1:]:
            print('###########################################Frame ID###############################')
            print(f_i)
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
                #print('###########################################translation##############################')
                #print('translation[:, 0]', translation[:, 0])

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))


    else:
        # Here we input all frames to the pose net (and predict all poses) together
        if selff.opt.pose_model_type in ["separate_resnet", "posecnn"]:
            pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in selff.opt.frame_ids if i != "s"], 1)
            for key, ipt in inputs.items():
                #inputs[key] = ipt.to(selff.device)
                #print(key)
#                for  i in selff.opt.frame_ids: 
#                    print('iiiiiiiiiiiiiiiiiii is: ' + str(i))
#                    
#                    if i != "s":
#                        pose_inputs = torch.cat([inputs[("color_aug", i, 0)]], 1)
#                        #pose_inputs = torch.cat(pose_inputs, 1)

            if selff.opt.pose_model_type == "separate_resnet":
                pose_inputs = [selff.models["pose_encoder"](pose_inputs)]

        elif selff.opt.pose_model_type == "shared":
            pose_inputs = [features[i] for i in selff.opt.frame_ids if i != "s"]

        axisangle, translation = selff.models["pose"](pose_inputs)
        #print(pose_inputs)
        for i, f_i in enumerate(selff.opt.frame_ids[1:]):
            if f_i != "s":
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])
                
    print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' + str(selff.opt.frame_ids))        
    print(selff.opt.pose_model_type)
    print(axisangle.size())
    print(translation.size())
    return outputs

def val(self):
    """Validate the model on a single minibatch
    """
    selff.set_eval()
    try:
        inputs = selff.val_iter.next()
    except StopIteration:
        selff.val_iter = iter(selff.val_loader)
        inputs = selff.val_iter.next()

    with torch.no_grad():
        outputs, losses = selff.process_batch(inputs)

        if "depth_gt" in inputs:
            selff.compute_depth_losses(inputs, outputs, losses)

        selff.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    selff.set_train()

def generate_images_pred(self, inputs, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    for scale in selff.opt.scales:
        disp = outputs[("disp", scale)]
        if selff.opt.v1_multiscale:
            source_scale = scale
        else:
            disp = F.interpolate(
                disp, [selff.opt.height, selff.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

        _, depth = disp_to_depth(disp, selff.opt.min_depth, selff.opt.max_depth)

        outputs[("depth", 0, scale)] = depth

        for i, frame_id in enumerate(selff.opt.frame_ids[1:]):

            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            # from the authors of https://arxiv.org/abs/1712.00175
            if (selff.opt.pose_model_type == "posecnn") and frame_id != "s":

                #if frame_id != "s":
                #print(outputs[("axisangle", 0, 0)])
                #print(outputs)

                axisangle = outputs[("axisangle", 0, frame_id)]
                translation = outputs[("translation", 0, frame_id)]

                inv_depth = 1 / depth
                mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
#                    pred_poses = []
#                    pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().detach().numpy())
#                    #print('###########################################translation##############################')
#                    #print(pred_poses)
                T = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

            cam_points = selff.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = selff.project_3d[source_scale](
                cam_points, inputs[("K", source_scale)], T)

            outputs[("sample", frame_id, scale)] = pix_coords

            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")

            if not selff.opt.disable_automasking:
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

def compute_reprojection_loss(self, pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if selff.opt.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = selff.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def compute_losses(self, inputs, outputs):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    losses = {}
    total_loss = 0

    for scale in selff.opt.scales:
        loss = 0
        reprojection_losses = []

        if selff.opt.v1_multiscale:
            source_scale = scale
        else:
            source_scale = 0

        disp = outputs[("disp", scale)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in selff.opt.frame_ids[1:]:
            pred = outputs[("color", frame_id, scale)]
            reprojection_losses.append(selff.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not selff.opt.disable_automasking:
            identity_reprojection_losses = []
            for frame_id in selff.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    selff.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if selff.opt.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

        elif selff.opt.predictive_mask:
            # use the predicted mask
            mask = outputs["predictive_mask"]["disp", scale]
            if not selff.opt.v1_multiscale:
                mask = F.interpolate(
                    mask, [selff.opt.height, selff.opt.width],
                    mode="bilinear", align_corners=False)

            reprojection_losses *= mask

            # add a loss pushing mask to 1 (using nn.BCELoss for stability)
            weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            loss += weighting_loss.mean()

        if selff.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not selff.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if not selff.opt.disable_automasking:
            outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += selff.opt.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= selff.num_scales
    losses["loss"] = total_loss
    return losses

def compute_depth_losses(self, inputs, outputs, losses):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    depth_pred = outputs[("depth", 0, 0)]
    depth_pred = torch.clamp(F.interpolate(
        depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    for i, metric in enumerate(selff.depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu())


def log_time(self, batch_idx, duration, loss):
    """Print a logging statement to the terminal
    """
    samples_per_sec = selff.opt.batch_size / duration
    time_sofar = time.time() - selff.start_time
    training_time_left = (
        selff.num_total_steps / selff.step - 1.0) * time_sofar if selff.step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
        " | loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(selff.epoch, batch_idx, samples_per_sec, loss,
                              sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

def log(self, mode, inputs, outputs, losses):
    """Write an event to the tensorboard events file
    """
    writer = selff.writers[mode]
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, selff.step)

    for j in range(min(4, selff.opt.batch_size)):  # write a maxmimum of four images
        for s in selff.opt.scales:
            for frame_id in selff.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, selff.step)
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, selff.step)

            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs[("disp", s)][j]), selff.step)

            if selff.opt.predictive_mask:
                for f_idx, frame_id in enumerate(selff.opt.frame_ids[1:]):
                    writer.add_image(
                        "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                        outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                        selff.step)

            elif not selff.opt.disable_automasking:
                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], selff.step)

def save_opts(self):
    """Save options to disk so we know what we ran this experiment with
    """
    models_dir = os.path.join(selff.log_path, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    to_save = selff.opt.__dict__.copy()

    with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)

def save_model(self):
    """Save model weights to disk
    """
    save_folder = os.path.join(selff.log_path, "models", "weights_{}".format(selff.epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in selff.models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == 'encoder':
            # save the sizes - these are needed at prediction time
            to_save['height'] = selff.opt.height
            to_save['width'] = selff.opt.width
            to_save['use_stereo'] = selff.opt.use_stereo
        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(selff.model_optimizer.state_dict(), save_path)

def load_model(self):
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
        selff.model_optimizer.load_state_dict(optimizer_dict)
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")
