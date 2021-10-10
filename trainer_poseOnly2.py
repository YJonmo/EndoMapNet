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

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

    # from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
        print(self.device)
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

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                
                if self.opt.use_pose=='1':
                    self.models["pose_encoder2"] = networks.ResnetEncoder(
                        self.opt.num_layers,
                        self.opt.weights_init == "pretrained",
                        num_input_images=self.num_pose_frames)
                    self.models["pose_encoder2"].to(self.device)
                    self.parameters_to_train2 = []
                    self.parameters_to_train2 += list(self.models["pose_encoder2"].parameters())
                    
                    self.models["pose2"] = networks.PoseDecoder(
                    self.models["pose_encoder2"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=self.num_frames_to_predict_for)
                    self.models["pose2"].to(self.device)
                    self.parameters_to_train2 += list(self.models["pose2"].parameters())
                    
                        
            elif self.opt.pose_model_type == "posecnn":
               if self.opt.use_pose=='1':
                    self.models["pose2"] = networks.PoseCNN(
                        self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                    self.models["pose2"].to(self.device)
                    self.parameters_to_train2 = []
                    self.parameters_to_train2 += list(self.models["pose2"].parameters())

        

        if self.opt.use_pose=='1':
            
            if self.opt.pose2_loss=="L2":
                self.criterion = torch.nn.MSELoss()
            else:
                self.criterion = torch.nn.L1Loss()
                #self.criterion = torch.nn.CrossEntropyLoss()
                #self.criterion = torch.nn.PoissonNLLLoss()
                
            if self.opt.pose2_optim=="SGD":
                self.model_optimizer2 = optim.SGD(self.parameters_to_train2, self.opt.learning_rate*5, momentum=0.9, weight_decay=self.opt.learning_rate/10)

            else:    
                self.model_optimizer2 = optim.Adam(self.parameters_to_train2, self.opt.learning_rate)
           
            self.model_lr_scheduler2 = optim.lr_scheduler.StepLR(
                self.model_optimizer2, self.opt.scheduler_step_size, 0.1)

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
        if self.opt.use_pose=='1':
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


        print("Using split:\n  ", self.opt.split)
        #print("There are {:d} training items and {:d} validation items\n".format(
        #    len(train_dataset), len(val_dataset)))
        print("There are {:d} training items\n".format(
            len(train_dataset)))
                           
        
        self.save_opts()

        self.opt.trans_weight=torch.from_numpy(np.array(self.opt.trans_weight))
        self.opt.trans_weight = self.opt.trans_weight.to(self.device)
        self.last_loss = self.criterion(torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
        self.last_loss = self.last_loss.to(self.device)

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
        if self.opt.use_pose=='1': 
            self.model_lr_scheduler2.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx) + str('   self.epoch ')+ str(self.epoch))
            #print(inputs)
            before_op_time = time.time()


            losses={}
            if self.opt.use_pose=='1':
                outputs2, losses2 = self.process_batch2(inputs)
                self.model_optimizer2.zero_grad()   # clear the buffer
                if not(torch.isnan(losses2["loss2"])):
                    
                    losses2["loss2"].backward()           # back propagate the loss                   
                    self.model_optimizer2.step()        # update the weights
                    losses["loss2"] = losses2["loss2"]
                    losses["loss2_Normal"] = losses2["loss2_Normal"]
                    losses["loss2_Raw"] = losses2["loss2_Raw"] 
                    losses["loss2_Normal_Ang"] = losses2["loss2_Normal_Ang"]
                    losses["loss2_Raw_Ang"] = losses2["loss2_Raw_Ang"]  
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 300
            late_phase = self.step % 300 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss2"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs2, losses)
                if self.opt.use_pose=='1':
                    self.val_pose()
                    
            self.step += 1


    def process_batch2(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            #print(key)
        #print('11111111111111111111111111111111111111111111111111111111111111111111111111inputs[("translation", frame_id, 0)]')
        #print(inputs[("translation", 1, 0)])
        outputs2 = {}
        if self.use_pose_net:
            #outputs.update(self.predict_poses(inputs, features))
            #outputs = {}
            if self.num_pose_frames == 2:
                # In this setting, we compute the pose to each source frame via a
                # separate forward pass through the pose network.
        
                # select what features the pose network takes as input
                if self.opt.pose_model_type == "shared": # if shared is used then the features are the same as features are obtained from the depth encoder
                    print('Error')		
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

 
    
    
    


        frame_lengh = 0
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                frame_lengh +=1   
        translation_gt = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=True, dtype=torch.float64) 
        translation_gt = translation_gt.to(device='cuda')
        translation_pred = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=False, dtype=torch.float64) 
        translation_pred = translation_pred.to(device='cuda')
        
        axisangle_gt = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=True, dtype=torch.float64) 
        axisangle_gt = axisangle_gt.to(device='cuda')        
        axisangle_pred = torch.zeros(self.opt.batch_size, 3, frame_lengh, requires_grad=False, dtype=torch.float64) 
        axisangle_pred = axisangle_pred.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float64) 
        i=0
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                translation_pred[:, :, i] = outputs2[("translation", 0, frame_id)][:,:]*100
                axisangle_pred[:, :, i] = outputs2[("axisangle", 0, frame_id)][:,:]*100
                axisangle_gt[:, :, i] = inputs[("axisangle", frame_id, 0)]*(1/1)
                translation_gt[:, :, i] = inputs[("translation", frame_id, 0)]*(1/1)

                i += 1
        print('axisangle_gt[0,:,i-1].abs().mean(): ')                
        print(axisangle_gt[0,:,i-1].data) 
#        print('axisangle_pred[0,:,i].abs().mean(): ')
#        print(axisangle_pred[0,:,i-1].data) 
#        
#        print('translation_gt[0,:,i].abs().mean(): ')                
#        print(translation_gt[0,:,i-1].data) 
#        print('translation_pred[0,:,i].abs().mean(): ')
#        print(translation_pred[0,:,i-1].data) 

        for Bach_ind in range(translation_gt.size()[0]):
            for Frame_ind in range(translation_gt.size()[2]):
                translation_gt[Bach_ind, :, Frame_ind] = translation_gt[Bach_ind, :, Frame_ind]*self.opt.trans_weight

        losses2 = {}



        
        

        
                    
        if  torch.mean(inputs[("translation", self.opt.frame_ids[1], 0)]) == 0:     # if the ground truth pose was not available then use the loss of the preveious run
            #losses2["loss2"] = self.last_loss.item() ;
            
            losses2["loss2_Normal"]=self.criterion(F.normalize(translation_pred), F.normalize(translation_pred))  * 1.0     
            losses2["loss2_Raw"] = self.criterion(translation_pred, translation_pred) 
    
            losses2["loss2_Normal_Ang"]=self.criterion(F.normalize(axisangle_pred), F.normalize(axisangle_pred))  * .2     
            losses2["loss2_Raw_Ang"] = self.criterion(axisangle_pred, axisangle_pred) * .5
            losses2["loss2"] = self.last_loss.item() + (losses2["loss2_Normal"] + losses2["loss2_Raw"] + losses2["loss2_Normal_Ang"] + losses2["loss2_Raw_Ang"])/(10000);

            
            
        else:
            losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt), F.normalize(translation_pred))  * 1.0     
            losses2["loss2_Raw"] = self.criterion(translation_gt, translation_pred) 
    
            losses2["loss2_Normal_Ang"]=self.criterion(F.normalize(axisangle_gt), F.normalize(axisangle_pred))  * .2    
            losses2["loss2_Raw_Ang"] = self.criterion(axisangle_gt, axisangle_pred) * 0.5
            losses2["loss2"] = (losses2["loss2_Normal"] + losses2["loss2_Raw"] + losses2["loss2_Normal_Ang"] + losses2["loss2_Raw_Ang"])
            losses2["loss2"] = (losses2["loss2_Normal"] + losses2["loss2_Raw"] + losses2["loss2_Normal_Ang"] + losses2["loss2_Raw_Ang"])
            
            self.last_loss = (losses2["loss2"] + self.last_loss)/2 

        print("loss2_Normal: " + str(losses2["loss2_Normal"].data))   
        print("loss2_Raw: "+ str(losses2["loss2_Raw"].data))  
        print("loss2_Normal_Ang: " + str(losses2["loss2_Normal_Ang"].data))   
        print("loss2_Raw_Ang_: "+ str(losses2["loss2_Raw_Ang"].data))  
        print("loss2: "+ str(losses2["loss2"].data))  
        print("lastLost: "+ str(self.last_loss.item()))  

        return outputs2, losses2





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
            outputs, losses = self.process_batch2(inputs)
            if not(torch.isnan(losses["loss2"])):
                    
                losses2 ={}
                losses2['val'] = losses['loss2']
                losses2['val_Normal'] = losses['loss2_Normal']
                losses2['val_Raw'] = losses['loss2_Raw']
                losses2['val_Normal_Ang'] = losses['loss2_Normal_Ang']
                losses2['val_Raw_Ang'] = losses['loss2_Raw_Ang']
                del losses
                writer = self.writers["train"]
                for l, v in losses2.items():
                    writer.add_scalar("{}".format(l), v, self.step)
                
                #self.log("val", inputs, outputs, losses)
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Pose val loss: ' + str(losses3['loss2'])) 
                del inputs, outputs, losses2
            
        self.set_train()

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
        print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs')
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)


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
        torch.save(self.model_optimizer2.state_dict(), save_path)

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
            self.model_optimizer2.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
