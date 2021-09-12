# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

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


class Evaluate:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        print(self.device)
        self.num_input_frames = len(self.opt.frame_ids)
        self.opt.frame_ids_sorted = []
        for i in self.opt.frame_ids: 
            self.opt.frame_ids_sorted.append(i)
        self.opt.frame_ids_sorted.sort()
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        if self.opt.pose_model_input == "pairs":
            self.num_frames_to_predict_for = 2
        elif (self.opt.pose_model_input == "all") and (self.opt.use_stereo):
            self.num_frames_to_predict_for = self.num_input_frames - 1 
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s") # this gives the id 's' to  the other image and it means it is a stereo pair


        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":


                self.models["pose_encoder2"] = networks.ResnetEncoder(
                self.opt.num_layers,
                False,
                num_input_images=self.num_pose_frames)
                self.models["pose_encoder2"].to(self.device)

                
                self.models["pose2"] = networks.PoseDecoder(
                self.models["pose_encoder2"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=self.num_frames_to_predict_for)
                self.models["pose2"].to(self.device)


            elif self.opt.pose_model_type == "posecnn":

                self.models["pose2"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                self.models["pose2"].to(self.device)
                self.parameters_to_train2 = []
                self.parameters_to_train2 += list(self.models["pose2"].parameters())


        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "Custom": datasets.Custom}
        self.dataset = datasets_dict[self.opt.dataset]



        print('self.opt.split: ' + self.opt.split)
        #fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "_files.txt")
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, self.opt.split+".txt")

        print('fpath fpath fpath: ' + fpath)
        print('os.path.dirname(__file__): ' + os.path.dirname(__file__))
        
        
        train_filenames = readlines(fpath.format("train"))
        #print(train_filenames)
        #val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs


        train_dataset = self.dataset(self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, use_pose=self.opt.use_pose)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)


        self.writers = {}

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def evaluate(self):
        """Run the entire evaluation pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        #self.opt.num_epochs = 1
        self.run_epoch()
            


    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.set_eval()
        self.pred_poses2 = []
        for batch_idx, inputs in enumerate(self.train_loader):
            if batch_idx == 0:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                    print(key)
            print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx))


          
            outputs2 = self.process_batch2(inputs)
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    self.pred_poses2.append((outputs2[("cam_T_cam", 0, f_i)]).cpu().detach().numpy())


        self.pred_poses2 = np.concatenate(self.pred_poses2)
        save_path = os.path.join(self.opt.load_weights_folder, "poses_" + self.opt.split[:]+".npy")
        np.save(save_path, self.pred_poses2)
        print("-> Predictions saved to", save_path)


    def process_batch2(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            #print(key)
        print('11111111111111111111111111111111111111111111111111111111111111111111111111inputs[("translation", frame_id, 0)]')


        #outputs.update(self.predict_poses(inputs, features))
        outputs2 = {}
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
                        pose_inputs2 = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs2 = [pose_feats[0], pose_feats[f_i]]
                        
                        
                
                        
                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs2 = [self.models["pose_encoder2"](torch.cat(pose_inputs2, 1))]

                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs2 = torch.cat(pose_inputs2, 1)

    
                    axisangle2, translation2 = self.models["pose2"](pose_inputs2)
                    if (f_i < 0):
                        translation2 = translation2*(-1)
                    outputs2[("axisangle", 0, f_i)] = axisangle2[:, 0, 0 , :]
                    outputs2[("translation", 0, f_i)] = translation2[:, 0, 0 , :]
                    
                    #translation_gt = inputs[("translation", f_i, 0)]/(0.0015/0.1)/1000 
                    translation_gt = inputs[("translation", f_i, 0)]*(.01/1)
                    print('\n translation_pred[:,0].data: ')
                    print((translation2[:,0].data)*100)   
                    print('translation_gt[:,0]: ')
                    print((translation_gt[:,:].data))    
                    print('\n ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ')
                    
                    outputs2[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle2[:, 0, : , :], translation2[:, 0, : , :], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs2 = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids_sorted if i != "s"], 1)
                
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs2 = [self.models["pose_encoder2"](pose_inputs2)]


            axisangle2, translation2 = self.models["pose2"](pose_inputs2)
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs2[("axisangle", 0, f_i)] = axisangle2[:, i, 0 , :]
                    outputs2[("translation", 0, f_i)] = translation2[:, i, 0 , :]
                    outputs2[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle2[:, i, : , :], translation2[:, i, : , :])
        return outputs2



    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
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
            #self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
