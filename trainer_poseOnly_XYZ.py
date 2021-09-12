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
#                    self.parameters_to_train2 = []
#                    self.parameters_to_train2 += list(self.models["pose_encoder2"].parameters())
                    
                    self.parameters_to_trainX = []
                    self.models["X"] = networks.PoseDecoderSingle(
                    self.models["pose_encoder2"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=self.num_frames_to_predict_for)
                    self.models["X"].to(self.device)
                    self.parameters_to_trainX += list(self.models["X"].parameters())

                    self.parameters_to_trainY = []
                    self.models["Y"] = networks.PoseDecoderSingle(
                    self.models["pose_encoder2"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=self.num_frames_to_predict_for)
                    self.models["Y"].to(self.device)
                    self.parameters_to_trainY += list(self.models["Y"].parameters())

                    self.parameters_to_trainZ = []
                    self.models["Z"] = networks.PoseDecoderSingle(
                    self.models["pose_encoder2"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=self.num_frames_to_predict_for)
                    self.models["Z"].to(self.device)
                    self.parameters_to_trainZ += list(self.models["Z"].parameters())
                        
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
                self.model_optimizerX = optim.SGD(self.parameters_to_trainX, self.opt.learning_rate*5, momentum=0.9, weight_decay=self.opt.learning_rate/10)
                self.model_optimizerY = optim.SGD(self.parameters_to_trainY, self.opt.learning_rate*5, momentum=0.9, weight_decay=self.opt.learning_rate/10)
                self.model_optimizerZ = optim.SGD(self.parameters_to_trainZ, self.opt.learning_rate*5, momentum=0.9, weight_decay=self.opt.learning_rate/10)

            else:    
                self.model_optimizerX = optim.Adam(self.parameters_to_trainX, self.opt.learning_rate)
                self.model_optimizerY = optim.Adam(self.parameters_to_trainY, self.opt.learning_rate)
                self.model_optimizerZ = optim.Adam(self.parameters_to_trainZ, self.opt.learning_rate)
           
            self.model_lr_schedulerX = optim.lr_scheduler.StepLR(
                self.model_optimizerX, self.opt.scheduler_step_size, 0.1)
            self.model_lr_schedulerY = optim.lr_scheduler.StepLR(
                self.model_optimizerY, self.opt.scheduler_step_size, 0.1)
            self.model_lr_schedulerZ = optim.lr_scheduler.StepLR(
                self.model_optimizerZ, self.opt.scheduler_step_size, 0.1)
            

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
            self.model_lr_schedulerX.step()
            self.model_lr_schedulerY.step()
            self.model_lr_schedulerZ.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            print('batch_idxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ' + str(batch_idx) + str('   self.epoch ')+ str(self.epoch))
            #print(inputs)
            before_op_time = time.time()


            losses={}

            Pose_features = self.getFeatures(inputs)




            outputsX, lossesX = self.process_batchX(inputs, Pose_features)
            self.model_optimizerX.zero_grad()   # clear the buffer
            if not(torch.isnan(lossesX["loss2"])):
                lossesX["loss2"].backward()           # back propagate the loss                   
                self.model_optimizerX.step()        # update the weights

#            outputsX, lossesX = self.process_batchX(inputs, Pose_features)
#            self.model_optimizerX.zero_grad()   # clear the buffer
#            if not(torch.isnan(lossesX["loss2"])):
#                lossesX["loss2"].backward()           # back propagate the loss                   
#                self.model_optimizerX.step()        # update the weights
#
#                
#            outputsZ, lossesZ = self.process_batchZ(inputs, Pose_features)
#            self.model_optimizerZ.zero_grad()   # clear the buffer
#            if not(torch.isnan(lossesZ["loss2"])):
#                lossesZ["loss2"].backward()           # back propagate the loss                   
#                self.model_optimizerZ.step()        # update the weights           
                
            duration = time.time() - before_op_time

            losses["loss2X"] = lossesX["loss2"]
            losses["loss2_NormalX"] = lossesX["loss2_Normal"]
            losses["loss2_RawX"] = lossesX["loss2_Raw"] 
#            losses["loss2Y"] = lossesX["loss2"]
#            losses["loss2_NormalY"] = lossesX["loss2_Normal"]
#            losses["loss2_RawY"] = lossesX["loss2_Raw"] 
#            losses["loss2Z"] = lossesX["loss2"]
#            losses["loss2_NormalZ"] = lossesX["loss2_Normal"]
#            losses["loss2_RawZ"] = lossesX["loss2_Raw"] 

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 50
            late_phase = self.step % 50 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss2X"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputsX, losses)
                if self.opt.use_pose=='1':
                    self.val_pose()
                    
            self.step += 1


    def getFeatures(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            #print(key)
        #print('11111111111111111111111111111111111111111111111111111111111111111111111111inputs[("translation", frame_id, 0)]')
        #print(inputs[("translation", 1, 0)])
        if self.use_pose_net:
            #outputs.update(self.predict_poses(inputs, features))
            #outputs = {}



            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs2 = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids_sorted if i != "s"], 1)
                
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs2 = [self.models["pose_encoder2"](pose_inputs2)]

        return pose_inputs2




    def process_batchX(self, inputs, pose_inputs2):
        outputs2 = {}
        translation2 = self.models["X"](pose_inputs2)
        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            if f_i != "s":
                outputs2[("X", 0, f_i)] = translation2[:, i, 0, :]
        
        frame_lengh = 0        
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                frame_lengh +=1   
        translation_gt = torch.zeros(self.opt.batch_size, frame_lengh, requires_grad=True, dtype=torch.float64) 
        translation_gt = translation_gt.to(device='cuda')
        translation_pred = torch.zeros(self.opt.batch_size, frame_lengh, requires_grad=False, dtype=torch.float64) 
        translation_pred = translation_pred.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float64) 
        i=0
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                #axisangle_pred = outputs[("axisangle", 0, frame_id)]

                translation_pred[:,  i] = outputs2[("X", 0, frame_id)][:,0]*100
                #axisangle_gt = inputs[("axisangle", frame_id, 0)]
                translation_gt[:,  i] = inputs[("translation", frame_id, 0)][:,0]
                i += 1
            
        print('\n translation_pred[0,:,i].abs().mean(): ')
        print(translation_pred[0,i-1].data) 
        print('translation_gt[0,:,i].abs().mean(): ')
        print(translation_gt[0,i-1].data)        


        losses2 = {}
        losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt), F.normalize(translation_pred))  * 1.5     
        losses2["loss2_Raw"] = self.criterion(translation_gt, translation_pred) 
        print("loss2_Normal: " + str(losses2["loss2_Normal"].data))   
        print("loss2_Raw: "+ str(losses2["loss2_Raw"].data))  
        #losses2["loss2"] = abs_diff_trans.mean() * coeff
        #losses2["loss2"] = abs_diff_trans.abs().mean() * coeff
        #losses2["loss2"] = self.criterion(translation_pred[:],translation_gt[:])  * coeff
        losses2["loss2"] = (losses2["loss2_Normal"] + losses2["loss2_Raw"])/frame_lengh

        return outputs2, losses2



                        
                        
    def process_batchY(self,  inputs, pose_inputs2):
        outputs2 = {}
        translation2 = self.models["Y"](pose_inputs2)
        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            if f_i != "s":
                outputs2[("Y", 0, f_i)] = translation2[:, i, 0, :]
        
        frame_lengh = 0        
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                frame_lengh +=1   
        translation_gt = torch.zeros(self.opt.batch_size, frame_lengh, requires_grad=True, dtype=torch.float64) 
        translation_gt = translation_gt.to(device='cuda')
        translation_pred = torch.zeros(self.opt.batch_size, frame_lengh, requires_grad=False, dtype=torch.float64) 
        translation_pred = translation_pred.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float64) 
        i=0
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                #axisangle_pred = outputs[("axisangle", 0, frame_id)]

                translation_pred[:,  i] = outputs2[("Y", 0, frame_id)][:,0]*100
                #axisangle_gt = inputs[("axisangle", frame_id, 0)]
                translation_gt[:,  i] = inputs[("translation", frame_id, 0)][:,0]
                i += 1
            
        print('\n translation_pred[0,:,i].abs().mean(): ')
        print(translation_pred[0,i-1].data) 
        print('translation_gt[0,:,i].abs().mean(): ')
        print(translation_gt[0,i-1].data)        


        losses2 = {}
        losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt), F.normalize(translation_pred))  * 1.5     
        losses2["loss2_Raw"] = self.criterion(translation_gt, translation_pred) 
        print("loss2_Normal: " + str(losses2["loss2_Normal"].data))   
        print("loss2_Raw: "+ str(losses2["loss2_Raw"].data))  
        #losses2["loss2"] = abs_diff_trans.mean() * coeff
        #losses2["loss2"] = abs_diff_trans.abs().mean() * coeff
        #losses2["loss2"] = self.criterion(translation_pred[:],translation_gt[:])  * coeff
        losses2["loss2"] = (losses2["loss2_Normal"] + losses2["loss2_Raw"])/frame_lengh

        return outputs2, losses2


                  
                        
    def process_batchZ(self,  inputs, pose_inputs2):
        outputs2 = {}
        translation2 = self.models["Z"](pose_inputs2)
        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            if f_i != "s":
                outputs2[("Z", 0, f_i)] = translation2[:, i, 0, :]
        
        frame_lengh = 0        
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                frame_lengh +=1   
        translation_gt = torch.zeros(self.opt.batch_size, frame_lengh, requires_grad=True, dtype=torch.float64) 
        translation_gt = translation_gt.to(device='cuda')
        translation_pred = torch.zeros(self.opt.batch_size, frame_lengh, requires_grad=False, dtype=torch.float64) 
        translation_pred = translation_pred.to(device='cuda')
        #abs_diff_axang = torch.zeros(i, dtype=torch.float64) 
        i=0
        for frame_id in (self.opt.frame_ids[1:]):
            if frame_id != "s":
                #axisangle_pred = outputs[("axisangle", 0, frame_id)]

                translation_pred[:,  i] = outputs2[("Z", 0, frame_id)][:,0]*100
                #axisangle_gt = inputs[("axisangle", frame_id, 0)]
                translation_gt[:,  i] = inputs[("translation", frame_id, 0)][:,0]
                i += 1
            
        print('\n translation_pred[0,:,i].abs().mean(): ')
        print(translation_pred[0,i-1].data) 
        print('translation_gt[0,:,i].abs().mean(): ')
        print(translation_gt[0,i-1].data)        


        losses2 = {}
        losses2["loss2_Normal"]=self.criterion(F.normalize(translation_gt), F.normalize(translation_pred))  * 1.5     
        losses2["loss2_Raw"] = self.criterion(translation_gt, translation_pred) 
        print("loss2_Normal: " + str(losses2["loss2_Normal"].data))   
        print("loss2_Raw: "+ str(losses2["loss2_Raw"].data))  
        #losses2["loss2"] = abs_diff_trans.mean() * coeff
        #losses2["loss2"] = abs_diff_trans.abs().mean() * coeff
        #losses2["loss2"] = self.criterion(translation_pred[:],translation_gt[:])  * coeff
        losses2["loss2"] = (losses2["loss2_Normal"] + losses2["loss2_Raw"])/frame_lengh

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
            
            Pose_features = self.getFeatures(inputs)
            outputsX, lossesX = self.process_batchX(inputs, Pose_features)

            
            losses2 ={}
            if not(torch.isnan(lossesX["loss2"])):
                    
                losses2['valX'] = lossesX['loss2']
                losses2['val_NormalX'] = lossesX['loss2_Normal']
                losses2['val_RawX'] = lossesX['loss2_Raw']
                del lossesX
                writer = self.writers["train"]
                for l, v in losses2.items():
                    writer.add_scalar("{}".format(l), v, self.step)

                #del outputX
            del inputs, outputsX, losses2, Pose_features

#            outputsY, lossesY = self.process_batchY(Pose_features)
#            
#            
#            if not(torch.isnan(lossesY["loss2"])):
#                    
#                losses2['valY'] = lossesY['loss2']
#                losses2['val_NormalY'] = lossesY['loss2_Normal']
#                losses2['val_RawY'] = lossesY['loss2_Raw']
#                del lossesY
#                writer = self.writers["train"]
#                for l, v in losses2.items():
#                    writer.add_scalar("{}".format(l), v, self.step)
#
#                del outputY, lossesY
# 
#            outputsZ, lossesZ = self.process_batchX(Pose_features)
#            
#            
#            if not(torch.isnan(lossesX["loss2"])):
#                    
#                losses2['valZ'] = lossesZ['loss2']
#                losses2['val_NormalZ'] = lossesZ['loss2_Normal']
#                losses2['val_RawZ'] = lossesZ['loss2_Raw']
#                del lossesZ
#                writer = self.writers["train"]
#                for l, v in losses2.items():
#                    writer.add_scalar("{}".format(l), v, self.step)

                #del inputs, outputZ, losses2, Pose_features
            
            
            
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
        torch.save(self.model_optimizerX.state_dict(), save_path)

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
