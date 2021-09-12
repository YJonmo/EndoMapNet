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
opts.log_dir="ScratchTraining2"
opts.split='Cadava5_3DPrint_Sheep_filt'
opts.dataset="Custom"
opts.png=True
opts.height=256
opts.width=256
opts.pose_model_input="pairs"
opts.num_epochs=150
opts.frame_ids=[0, 1, 5]
opts.batch_size=4
#opts.learning_rate=0.00002
opts.pose_model_type="separate_resnet"
opts.use_stereo=True
opts.disparity_smoothness=.005
opts.use_pose='1'
#opts.load_weights_folder='/home/jonmoham/Python/monodepth2-master/3D_MonoPose0-1-212_Pair_Bach12_Smooth.005_UsePose2/mdp/models/weights_10'
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
if selff.opt.pose_model_input == "pairs":
    selff.num_frames_to_predict_for = 2
elif (selff.opt.pose_model_input == "all") and (selff.opt.use_stereo):
    selff.num_frames_to_predict_for = selff.num_input_frames - 1 
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


        selff.models["pose_encoder2"] = networks.ResnetEncoder(
            selff.opt.num_layers,
            selff.opt.weights_init == "pretrained",
            num_input_images=selff.num_pose_frames)
        selff.models["pose_encoder2"].to(selff.device)
        selff.parameters_to_train2 = []
        selff.parameters_to_train2 += list(selff.models["pose_encoder2"].parameters())

        #selff.models["pose"] = networks.PoseDecoder(
        #    selff.models["pose_encoder"].num_ch_enc,
        #    num_input_features=1,
        #    num_frames_to_predict_for=2)
        selff.models["pose"] = networks.PoseDecoder(
            selff.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=selff.num_frames_to_predict_for)

        selff.models["pose2"] = networks.PoseDecoder(
            selff.models["pose_encoder2"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=selff.num_frames_to_predict_for)
        selff.models["pose2"].to(selff.device)
        selff.parameters_to_train2 += list(selff.models["pose2"].parameters())
        
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

selff.model_optimizer2 = optim.Adam(selff.parameters_to_train2, selff.opt.learning_rate)
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

if selff.opt.use_pose=='1':
    val_filenames = readlines(fpath.format("val"))
    val_dataset = selff.dataset(
        selff.opt.data_path, val_filenames, selff.opt.height, selff.opt.width,
        selff.opt.frame_ids, 4, is_train=False, img_ext=img_ext, use_pose=selff.opt.use_pose)
    selff.val_loader = DataLoader(
        val_dataset, selff.opt.batch_size, True,
        num_workers=selff.opt.num_workers, pin_memory=True, drop_last=True)
    selff.val_iter = iter(selff.val_loader)

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
selff.model_lr_scheduler2.step()

print("Training")
#%%selff.set_train()


"""Convert all models to training mode
"""
for m in selff.models.values():
    m.train()



#with torch.autograd.set_detect_anomaly(True):
for selff.epoch in range(selff.opt.num_epochs):
    #self.run_epoch()                
    for batch_idx, inputs in enumerate(selff.train_loader):
        
        print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx) + str('selff.epoch')+ str(selff.epoch))
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
            outputs2 = {}
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
                            pose_inputs2 = [pose_feats[f_i], pose_feats[0]]
                        else:
                            pose_inputs = [pose_feats[f_i], pose_feats[0]]
                            pose_inputs2 = [pose_feats[f_i], pose_feats[0]]
    
                        if selff.opt.pose_model_type == "separate_resnet":
                            pose_inputs = [selff.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            pose_inputs2 = [selff.models["pose_encoder2"](torch.cat(pose_inputs2, 1))]
    
                        elif selff.opt.pose_model_type == "posecnn":
                            pose_inputs = torch.cat(pose_inputs, 1)
    
                        axisangle, translation = selff.models["pose"](pose_inputs)
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
        
                        #selff.model_optimizer2.zero_grad()   # clear the buffer
                        axisangle2, translation2 = selff.models["pose2"](pose_inputs2)
                        outputs2[("axisangle", 0, f_i)] = axisangle2
                        outputs2[("translation", 0, f_i)] = translation2
    
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
                        
            #print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' + str(selff.opt.frame_ids))        
            #print(selff.opt.pose_model_type)
            #print(axisangle.size())
            #print(translation.size())
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
                
                
            total_loss /= selff.num_scales
            
            #total_loss += l1_loss_pose
            
            losses["loss"] = total_loss
            #return losses

    
        selff.model_optimizer.zero_grad()   # clear the buffer    
        #losses["loss"].backward(retain_graph=True)           # back propagate the loss 
        losses["loss"].backward()           # back propagate the loss 

    
        selff.model_optimizer.step()        # update the weights
        print('total_loss: ' + str(total_loss))
    
    
    #%%#######################################################loss 2 for pose translation##################################    
        #with torch.no_grad():
        i = 0
        for frame_id in (selff.opt.frame_ids[1:]):
            if frame_id != "s":
                i +=1   
        abs_diff_trans = torch.zeros(i, requires_grad=True, dtype=torch.float64) 
        abs_diff_trans = abs_diff_trans.to(device='cuda')
        translation_pred = torch.zeros(selff.opt.batch_size, 2, 1, 3, requires_grad=False, dtype=torch.float64) 
        translation_pred = translation_pred.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float64) 
        i=0
        for frame_id in (selff.opt.frame_ids[1:]):
            if frame_id != "s":
                #axisangle_pred = outputs[("axisangle", 0, frame_id)]
                #translation_pred = outputs2[("translation", 0, frame_id)]*1000
                translation_pred = outputs2[("translation", 0, frame_id)]*1000
                #axisangle_gt = inputs[("axisangle", frame_id, 0)]
                Sign = torch.sign(inputs[("translation", frame_id, 0)])
                translation_gt = Sign*torch.log(torch.abs(inputs[("translation", frame_id, 0)]/100))/100
                #translation_gt = inputs[("translation", frame_id, 0)]/(0.0015/0.1)/1000
                print('\n translation_pred[:,0].abs().mean(): ')
                print(translation_pred[:,0].data) 
                print('translation_gt[:,0].abs().mean(): ')
                print(translation_gt[:,:].data)    
                abs_diff_trans[i] = (translation_pred[:,0,0,:] - translation_gt[0,:] + 0.00001).abs().mean()
                #abs_diff_trans[i] = (translation_pred[:,0,0,:] - translation_gt[:,:]).square().sum().sqrt()
                #abs_diff_axang.append(torch.abs(axisangle_pred[:,0]   - axisangle_gt).abs().mean())
                i += 1
                               
    
        coeff = torch.zeros(1, requires_grad=True, dtype=torch.float64)  
        coeff = coeff.to(device='cuda')     
        coeff[0] = 1
        losses["loss2"] = abs_diff_trans.mean() * coeff
        #losses2 = outputs[("translation", 0, f_i)][:,0].mean()
        selff.model_optimizer2.zero_grad()   # clear the buffer    
        if not(torch.isnan(abs_diff_trans.mean())):
                
            #
            losses["loss2"].backward()           # back propagate the loss 
            selff.model_optimizer2.step()        # update the weights
            print('losses[loss2]: ' + str(losses["loss2"]*1))
    
    
        #selff.log("train", inputs, outputs, losses)
        #def log(self, mode, inputs, outputs, losses):
        if np.remainder(selff.step,30) == 0:
            #def val(self):
#            """Validate the model on a single minibatch
#            """
#            #selff.set_eval()
#            for m in selff.models.values():
#             m.eval()
#            
#            try:
#                inputs = selff.val_iter.next()
#            except StopIteration:
#                selff.val_iter = iter(selff.val_loader)
#                inputs = selff.val_iter.next()
#    
#            with torch.no_grad():
#                outputs, losses = selff.process_batch(inputs)
##    
##                if "depth_gt" in inputs:
##                    selff.compute_depth_losses(inputs, outputs, losses)
#    
#                selff.log("val", inputs, outputs, losses)
#                del inputs, outputs, losses
#    
#            #selff.set_train()
#            for m in selff.models.values():
#                m.train()
            
            
            
            
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
    
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), selff.step)

        
        
        
        
        
        
        selff.step += 1


#%%####################################################################################################################     
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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



