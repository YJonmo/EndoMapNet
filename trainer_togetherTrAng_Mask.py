# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
#from tensorboard import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
        print(self.device)
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.opt.frame_ids_sorted = []
        for i in self.opt.frame_ids: 
            self.opt.frame_ids_sorted.append(i)
        self.opt.frame_ids_sorted.sort()
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%self.num_pose_frames', self.num_pose_frames)
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        
        if self.opt.pose_model_input == "pairs":
            self.num_frames_to_predict_for = 2
        elif (self.opt.pose_model_input == "all") and (self.opt.use_stereo):
            self.num_frames_to_predict_for = self.num_input_frames - 1 
        
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s") # this gives the id 's' to  the other image and it means it is a stereo pair

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                print('LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL',self.num_pose_frames)
                print('self.models["pose_encoder"].num_ch_enc',self.models["pose_encoder"].num_ch_enc)
        
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        
#                if self.opt.use_pose=='1':
#                    self.models["pose_encoder2"] = networks.ResnetEncoder(
#                        self.opt.num_layers,
#                        self.opt.weights_init == "pretrained",
#                        num_input_images=self.num_pose_frames)
#                    self.models["pose_encoder2"].to(self.device)
#                    self.parameters_to_train2 = []
#                    self.parameters_to_train2 += list(self.models["pose_encoder2"].parameters())
#                    
#                    self.models["pose2"] = networks.PoseDecoder(
#                    self.models["pose_encoder2"].num_ch_enc,
#                        num_input_features=1,
#                        num_frames_to_predict_for=self.num_frames_to_predict_for)
#                    self.models["pose2"].to(self.device)
#                    self.parameters_to_train2 += list(self.models["pose2"].parameters())
            
        
                #self.models["pose"] = networks.PoseDecoder(
                #    self.models["pose_encoder"].num_ch_enc,
                #    num_input_features=1,
                #    num_frames_to_predict_for=2)
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=self.num_frames_to_predict_for)
        
                
            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)
        
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                
#                if self.opt.use_pose=='1':
#                    self.models["pose2"] = networks.PoseCNN(
#                        self.num_input_frames if self.opt.pose_model_input == "all" else 2)
#                    self.models["pose2"].to(self.device)
#                    self.parameters_to_train2 = []
#                    self.parameters_to_train2 += list(self.models["pose2"].parameters())

        
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        if self.opt.use_pose=='1':
            
            if self.opt.pose2_loss=="L2":
                self.criterion = torch.nn.MSELoss()
            else:
                self.criterion = torch.nn.L1Loss()
                #self.criterion = torch.nn.CrossEntropyLoss()
                #self.criterion = torch.nn.PoissonNLLLoss()
                
#            if self.opt.pose2_optim=="SGD":
#                self.model_optimizer2 = optim.SGD(self.parameters_to_train2, self.opt.learning_rate*5, momentum=0.9, weight_decay=self.opt.learning_rate/10)

#            else:    
#                self.model_optimizer2 = optim.Adam(self.parameters_to_train2, self.opt.learning_rate)
#           
#            self.model_lr_scheduler2 = optim.lr_scheduler.StepLR(
#                self.model_optimizer2, self.opt.scheduler_step_size, 0.1)
#
#        if self.opt.load_weights_folder is not None:
#            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "Custom": datasets.Custom}
        self.dataset = datasets_dict[self.opt.dataset]
        
        print('self.opt.split: ' + self.opt.split)
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        print('fpathfpathfpath: ' + fpath)
        print('os.path.dirname(__file__): ' + os.path.dirname(__file__))

        train_filenames = readlines(fpath.format("train"))
        
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

#        train_dataset = self.dataset(
#            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
#            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, use_pose=self.opt.use_pose)
        
        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        if 1==1:
            val_filenames = readlines(fpath.format("val"))
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, use_pose=self.opt.use_pose)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        #print("There are {:d} training items and {:d} validation items\n".format(
        #    len(train_dataset), len(val_dataset)))
        print("There are {:d} training items\n".format(
            len(train_dataset)))

        self.save_opts()
        #self.opt.trans_weight=torch.from_numpy(np.array(self.opt.trans_weight))
        self.opt.trans_weight=torch.Tensor(np.array(self.opt.trans_weight))
        self.opt.trans_weight = self.opt.trans_weight.to(self.device)
        if self.opt.use_pose=='1':
            self.last_loss = self.criterion(torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32))
            self.last_loss = self.last_loss.to(self.device)
            self.last_loss_val = self.criterion(torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32))
            self.last_loss_val = self.last_loss_val.to(self.device)
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
#        if self.opt.use_pose=='1': 
#            self.model_lr_scheduler2.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx) + str('   self.epoch ')+ str(self.epoch))
            #print(inputs)
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            #time.sleep(20.0)
#            if self.opt.use_pose=='1':
#                outputs2, losses2 = self.process_batch2(inputs)
#                self.model_optimizer2.zero_grad()   # clear the buffer
#                if not(torch.isnan(losses2["loss2"])):
#                    
#                    losses2["loss2"].backward()           # back propagate the loss                   
#                    self.model_optimizer2.step()        # update the weights
#                    losses["loss2"] = losses2["loss2"]
#                    losses["loss2_Normal"] = losses2["loss2_Normal"]
#                    losses["loss2_Raw"] = losses2["loss2_Raw"]            
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 3
            late_phase = self.step % 300 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #if self.opt.use_pose=='1':
                self.val_pose()
                    
            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            #print(key)
        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)


        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
            #print('###########################################Output###############################')
            #print(outputs)
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        #print('losses[loss]: ' + str(losses["loss"]))
        return outputs, losses



    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        pred_poses = []
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared": # if shared is used then the features are the same as features are obtained from the depth encoder
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:# if shared is not used then instead of feature, images are given to the pose decoder
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                #print('###########################################Pose_Feature###############################')
                #print(pose_feats)
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
#                    print('\n axisangle[:, 0].abs().mean(): ')
#                    print(axisangle[:, 0].data)

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids_sorted if i != "s"], 1)
#                for  i in self.opt.frame_ids: 
#                    print('iiiiiiiiiiiiiiiiiii is: ' + str(i))
#                    
#                    if i != "s":
#                        pose_inputs = torch.cat([inputs[("color_aug", i, 0)]], 1)
#                        #pose_inputs = torch.cat(pose_inputs, 1)

                #print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' + str(self.opt.frame_ids))        
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)
#            print(self.opt.pose_model_type)
#            print(axisangle.size())
#            print(translation.size())
            #print(pose_inputs)
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])
                    
        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def val_pose(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            #print('losssssssssssssssssssssssss')
            #print(losses)
            losses2 ={} 
            if self.opt.use_pose=='1':     
                if not(torch.isnan(losses["loss2"])):           
                    losses2['val'] = losses['loss2']
                    losses2['val_Normal'] = losses['loss2_Normal']
                    losses2['val_Normal_Ang'] = losses['loss2_Normal_Ang']
                    losses2['val_Normal_Mag'] = losses['loss2_Normal_Mag']
                    losses2['val_Normal_Ang_Mag'] = losses['loss2_Normal_Ang_Mag']
                    losses2["val_reprojection"] =  losses["loss_reprojection"]

                    writer = self.writers["train"]
                    for l, v in losses2.items():
                        writer.add_scalar("{}".format(l), v, self.step)
            else: 
                losses2["val_reprojection"] =  losses["loss"]
                   
                writer = self.writers["train"]
                for l, v in losses2.items():
                    writer.add_scalar("{}".format(l), v, self.step)
                
                #self.log("val", inputs, outputs, losses)
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Pose val loss: ' + str(losses3['loss2'])) 
                del inputs, outputs, losses2, losses
            
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if (self.opt.pose_model_type == "posecnn") and frame_id != "s":

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

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
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

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        #total_loss = torch.tensor([total_loss], dtype=torch.double)
        
        
        losses2 = {}
        losses2["loss2"] = 0
        if self.opt.use_pose=='1':
            frame_lengh = 0
            for frame_id in (self.opt.frame_ids[1:]):
                if frame_id != "s":
                    frame_lengh +=1   
            translation_gt = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=True, dtype=torch.float32) 
            translation_gt = translation_gt.to(device='cuda')
            translation_pred = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=False, dtype=torch.float32) 
            translation_pred = translation_pred.to(device='cuda')
            axisangle_gt = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=True, dtype=torch.float32) 
            axisangle_gt = axisangle_gt.to(device='cuda')
            axisangle_pred = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=False, dtype=torch.float32) 
            axisangle_pred = axisangle_pred.to(device='cuda')
            #abs_diff_axang = torch.zeros(i, dtype=torch.float32) 
            i=0
            for frame_id in (self.opt.frame_ids[1:]):
                if frame_id != "s":
                    #axisangle_pred = outputs[("axisangle", 0, frame_id)]
                    translation_pred[:, :, i] = outputs[("translation", 0, frame_id)][:, 0, 0 , :]*100
                    translation_gt[:, :, i] = inputs[("translation", frame_id, 0)]*(1/1)
                    axisangle_pred[:, :, i] = outputs[("axisangle", 0, frame_id)][:, 0, 0 , :]*100
                    axisangle_gt[:, :, i] = inputs[("axisangle", frame_id, 0)]*(1/1)
                    i += 1
            #print('\n translation_pred[0,:,i].abs().mean(): ')
            #print(translation_pred[0,:,i-1].data) 
            #print('axisangle_gt[0,:,i].abs().mean(): ')
            #print(axisangle_gt[0,:,i-1].data)    

            NonZeroBatch = 0
            for Bach_ind in range(translation_gt.size()[0]):
                if  torch.norm(translation_gt[Bach_ind,:, 0]) != 0:
                    NonZeroBatch +=1   
            if NonZeroBatch != 0:
                translation_gt2 = torch.zeros(NonZeroBatch, 3, frame_lengh, requires_grad=True, dtype=torch.float32) 
                translation_gt2 = translation_gt2.to(device='cuda')
                translation_pred2 = torch.zeros(NonZeroBatch, 3, frame_lengh, requires_grad=True, dtype=torch.float32) 
                translation_pred2 = translation_pred2.to(device='cuda')
                axisangle_gt2 = torch.zeros(NonZeroBatch, 3, frame_lengh, requires_grad=True, dtype=torch.float32) 
                axisangle_gt2 = axisangle_gt2.to(device='cuda')
                axisangle_pred2 = torch.zeros(NonZeroBatch, 3, frame_lengh, requires_grad=True, dtype=torch.float32) 
                axisangle_pred2 = axisangle_pred2.to(device='cuda')
                translation_GtMag = torch.zeros(NonZeroBatch, frame_lengh, requires_grad=True, dtype=torch.float32) 
                translation_GtMag = translation_GtMag.to(device='cuda')
                translation_PrMag = torch.zeros(NonZeroBatch, frame_lengh, requires_grad=True, dtype=torch.float32) 
                translation_PrMag = translation_PrMag.to(device='cuda')
                axisangle_GtMag = torch.zeros(NonZeroBatch, frame_lengh, requires_grad=True, dtype=torch.float32) 
                axisangle_GtMag = axisangle_GtMag.to(device='cuda')
                axisangle_PrMag = torch.zeros(NonZeroBatch, frame_lengh, requires_grad=True, dtype=torch.float32) 
                axisangle_PrMag = axisangle_PrMag.to(device='cuda')
                #Mask_pose = torch.zeros(self.opt.batch_size, 1, dtype=torch.bool)   
             
                NonZeroBatch = 0 		
                for Bach_ind in range(translation_gt.size()[0]):	
                    if  torch.norm(translation_gt[Bach_ind,:, 0]) != 0:
                        #translation_gt2[NonZeroBatch, :, :] = translation_gt[Bach_ind, :, :]
                        #translation_pred2[NonZeroBatch, :, :] = translation_pred[Bach_ind, :, :].clone()
                        axisangle_gt2[NonZeroBatch, :, :] = axisangle_gt[Bach_ind, :, :].clone()
                        axisangle_pred2[NonZeroBatch, :, :] = axisangle_pred[Bach_ind, :, :].clone()

                        for Frame_ind in range(translation_gt.size()[2]):
                    
                            translation_gt2[NonZeroBatch, :, Frame_ind]= translation_gt[Bach_ind, :, Frame_ind].clone()*self.opt.trans_weight
                            translation_pred2[NonZeroBatch, :, Frame_ind]= translation_pred[Bach_ind, :, Frame_ind].clone()*self.opt.trans_weight
                            translation_GtMag[NonZeroBatch,Frame_ind]=torch.norm(translation_gt2[NonZeroBatch,:,Frame_ind].clone(),dim=0)
                            translation_PrMag[NonZeroBatch, Frame_ind]= torch.norm(translation_pred2[NonZeroBatch, :, Frame_ind].clone(), dim=0)
                            axisangle_GtMag[NonZeroBatch, Frame_ind]= torch.norm(axisangle_gt2[NonZeroBatch, :, Frame_ind].clone(), dim=0)
                            axisangle_PrMag[NonZeroBatch, Frame_ind]= torch.norm(axisangle_pred2[NonZeroBatch,:,Frame_ind].clone(), dim=0) 

                        NonZeroBatch = NonZeroBatch + 1


                #Coeff =  15*frame_lengh
                Coeff =  150
                #losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt),F.normalize(translation_pred))/(15*frame_lengh) * 1.0     
                losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt2),F.normalize(translation_pred2))/(Coeff*1.5)     
                losses2["loss2_Normal_Mag"]=self.criterion(F.normalize(translation_GtMag, dim=0),F.normalize(translation_PrMag, dim=0))/Coeff      
                #losses2["loss2_Raw"] = self.criterion(translation_gt, translation_pred) 
                losses2["loss2_Normal_Ang"]=self.criterion(F.normalize(axisangle_gt2), F.normalize(axisangle_pred2))/(Coeff*2.5)  
                losses2["loss2_Normal_Ang_Mag"]=self.criterion(F.normalize(axisangle_GtMag, dim=0), F.normalize(axisangle_PrMag, dim=0))/(Coeff*2)  

                losses2["loss2"]=(losses2["loss2_Normal"]+losses2["loss2_Normal_Mag"]+losses2["loss2_Normal_Ang"]+losses2["loss2_Normal_Ang_Mag"])
            

                #print('It was hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1')             
                #print('translation_gt[:, :, 0]')
                #print(translation_gt[:, :, 0].data)
                #print('translation_gt2[:, :, 0]')
                #print(translation_gt2[:, :, 0].data)
                #print('translation_GtMag[:, 0]')
                #print(translation_GtMag[:, 0].data)
                #print('It was hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee2')

            else:                
                #Coeff =  15*frame_lengh
                Coeff =  150
                #losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt),F.normalize(translation_pred))/(15*frame_lengh) * 1.0     
                losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt),F.normalize(translation_pred))/Coeff     
                losses2["loss2_Normal_Mag"]=self.criterion(F.normalize(translation_gt, dim=0),F.normalize(translation_pred, dim=0))/Coeff      
                #losses2["loss2_Raw"] = self.criterion(translation_gt, translation_pred) 
                losses2["loss2_Normal_Ang"]=self.criterion(F.normalize(axisangle_gt), F.normalize(axisangle_pred))/(Coeff*2)  
                losses2["loss2_Normal_Ang_Mag"]=self.criterion(F.normalize(axisangle_gt, dim=0), F.normalize(axisangle_pred, dim=0))/(Coeff*2)  
                losses2["loss2"]=(losses2["loss2_Normal"]+losses2["loss2_Normal_Mag"]+losses2["loss2_Normal_Ang"]+losses2["loss2_Normal_Ang_Mag"])
    
                losses2["loss2"] = self.last_loss.item() + losses2["loss2_Normal"]*.00001;


            del translation_gt, translation_pred, axisangle_gt, axisangle_pred


            self.last_loss = (losses2["loss2"] + self.last_loss)/2 
              

            #print('loss2_Normal : ' + str(losses2["loss2_Normal"].data))
            #print('loss2_Normal_Ang : ' + str(losses2["loss2_Normal_Ang"].data))
            #print('loss2_Normal_Mag : ' + str(losses2["loss2_Normal_Mag"]))
            #print('loss2_Normal_Ang_Mag : ' + str(losses2["loss2_Normal_Ang_Mag"]))

   
                
            #losses2["loss2"] = losses2["loss2_Normal"] #+ losses2["loss2_Normal"]*(torch.norm(translation_gt)/frame_lengh)*2 #/frame_lengh #+ losses2["loss2_Raw"])/frame_lengh  
                   
       
            losses["loss_reprojection"] = total_loss
            losses["loss2_Normal"] = losses2["loss2_Normal"]
            losses["loss2_Normal_Ang"] = losses2["loss2_Normal_Ang"]
            losses["loss2_Normal_Mag"] = losses2["loss2_Normal_Mag"]
            losses["loss2_Normal_Ang_Mag"] = losses2["loss2_Normal_Ang_Mag"]
            losses["loss2"] = losses2["loss2"]
            print('######################################This epoch start #############################')
            print('Reprojection loss: ' + str(total_loss.data))
            print('self.last_loss: '    + str(self.last_loss.data))
            print('loss2: ' + str(losses2["loss2"]))   
            
            print("loss2_Normal: " + str(losses2["loss2_Normal"]))   
            print("loss2_Normal_Mag: " + str(losses2["loss2_Normal_Mag"]))   

            print("loss2_Normal_Ang: " + str(losses2["loss2_Normal_Ang"]))   
            print("loss2_Normal_Ang_Mag: " + str(losses2["loss2_Normal_Ang_Mag"]))   
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ End ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            
        #print('Reprojection loss: ' + str(total_loss.data))
        losses["loss"] = (total_loss + losses2["loss2"])
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

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(loss)
        
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


