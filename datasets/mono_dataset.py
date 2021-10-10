# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import math
import sys
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png',
                 use_pose=False,
                 use_quat=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        self.use_pose=use_pose
        self.use_quat=use_quat
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def euler2world(self, theta) :
        #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]], np.float64)                  
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]], np.float64)              
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]], np.float64)                                 
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def world2euler(self, World):   # best so far.
        tol = sys.float_info.epsilon * 10
        Euler = np.zeros((3), np.float64)
        #https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/  
        if abs(World.item(0,0))< tol and abs(World.item(1,0)) < tol:
           Euler[2] = 0
           Euler[1] = math.atan2(-World.item(2,0), World.item(0,0))
           Euler[0] = math.atan2(-World.item(1,2), World.item(1,1))
        else:   
           Euler[2] = math.atan2(World.item(1,0),World.item(0,0))
           sp = math.sin(Euler[0])
           cp = math.cos(Euler[0])
           Euler[1] = math.atan2(-World.item(2,0),cp*World.item(0,0)+sp*World.item(1,0))
           Euler[0] = math.atan2(sp*World.item(0,2)-cp*World.item(1,2),cp*World.item(1,1)-sp*World.item(0,1))
    
        return Euler  
    
    
    def world2quatr(self, World):                # 
        tr = World[0][0] + World[1][1] + World[2][2]
        if (tr > 0):
            S = np.sqrt(tr+1.0) * 2# // S=4*qw
            qw = 0.25 * S;
            qx = (World[2][1] - World[1][2]) / S;
            qy = (World[0][2] - World[2][0]) / S; 
            qz = (World[1][0] - World[0][1]) / S;
        elif ((World[0][0] > World[1][1])and(World[0][0] > World[2][2])):
            S = np.sqrt(1.0 + World[0][0] - World[1][1] - World[2][2]) * 2; # S=4*qx
            qw = (World[2][1] - World[1][2]) / S;
            qx = 0.25 * S;
            qy = (World[0][1] + World[1][0]) / S; 
            qz = (World[0][2] + World[2][0]) / S; 
        elif (World[1][1] > World[2][2]):
            S = np.sqrt(1.0 + World[1][1] - World[0][0] - World[2][2]) * 2; # S=4*qy
            qw = (World[0][2] - World[2][0]) / S;
            qx = (World[0][1] + World[1][0]) / S;
            qy = 0.25 * S;
            qz = (World[1][2] + World[2][1]) / S;
        else:
            S = np.sqrt(1.0 + World[2][2] - World[0][0] - World[1][1]) * 2; # S=4*qz
            qw = (World[1][0] - World[0][1]) / S;
            qx = (World[0][2] + World[2][0]) / S;
            qy = (World[1][2] + World[2][1]) / S;
            qz = 0.25 * S;
        quaternion=np.array([qw, qx, qy, qz]) 
        return   quaternion 

    def quatr2world(self, Coordinates):         #Coordinates is the quaternion
        W_O= np.float64(Coordinates[0])
        X_O= np.float64(Coordinates[1])
        Y_O= np.float64(Coordinates[2])
        Z_O= np.float64(Coordinates[3])
        World=np.zeros((4,4), np.float64)
        World[0][0]= 1 - 2*Y_O*Y_O - 2*Z_O*Z_O
        World[0][1]= 2*X_O*Y_O + 2*W_O*Z_O
        World[0][2]= 2*X_O*Z_O - 2*W_O*Y_O
        World[1][0]= 2*X_O*Y_O - 2*W_O*Z_O
        World[1][1]= 1 - 2*X_O*X_O - 2*Z_O*Z_O
        World[1][2]= 2*Y_O*Z_O + 2*W_O*X_O
        World[2][0]= 2*X_O*Z_O + 2*W_O*Y_O
        World[2][1]= 2*Y_O*Z_O - 2*W_O*X_O
        World[2][2]= 1 - 2*X_O*X_O - 2*Y_O*Y_O
        World[3][3]= 1
        World[0:3, 0:3] = np.transpose(World[0:3,0:3])
        return World
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        do_flip=False
        line = self.filenames[index].split()
        #print('folder: ' + line[0])
        folder = line[0]

        if (len(line) == 3) or (len(line) == 3+6) or (len(line) == 3+7):
            frame_index = int(line[1])
        else:
            frame_index = 0

        if (len(line) == 3) or (len(line) == 3+6) or (len(line) == 3+7):
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
#                print(folder)
#                print(frame_index)
#                print('folder is: ' + folder)
#                print('frame_index is: ' + str(frame_index + i))
#                print('side is: ' + side)
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            #side_sign = -1 if side == "l" else 1
            side_sign = 1 if side == "l" else -1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            #print('side_sign * baseline_sign' + str(side_sign * baseline_sign))
            #stereo_T[0, 3] = side_sign * baseline_sign * 0.00153            
            inputs["stereo_T"] = torch.from_numpy(stereo_T)


        if self.use_pose=='1':
        #if 1==1:
            Ref_Trans = np.array([np.float(line[3]), np.float(line[4]), np.float(line[5])])
            if self.use_quat=='1':
                Ref_Quatr = np.array([np.float(line[6]), np.float(line[7]), np.float(line[8]), np.float(line[9])], np.float64)
                #print(Ref_Euler)
                Ref_World = self.quatr2world(Ref_Quatr)
            else:
                Ref_Euler = np.array([np.float(line[6]), np.float(line[7]), np.float(line[8])], np.float64)
                #print(Ref_Euler)
                Ref_World = self.euler2world(Ref_Euler)
            for i in self.frame_idxs[1:]:
                if i != "s":
                    if self.use_quat=='1':
                        
                        if (index+i >= 0) and (index+i < len(self.filenames)):
                            Delta_line = self.filenames[index+i].split()
                            if  (Delta_line[0] == line[0]):
                                Delta = np.array([np.float(Delta_line[3]), np.float(Delta_line[4]), np.float(Delta_line[5])]) - Ref_Trans
#                                print('Delta_line: ')
#                                print(Delta_line)
                                Next_Quatr = np.array([np.float(Delta_line[6]), np.float(Delta_line[7]), np.float(Delta_line[8]), np.float(Delta_line[9])], np.float64)
                                Next_World = self.quatr2world(Next_Quatr)
                                Delta_World = np.dot(Ref_World, np.linalg.pinv(Next_World))
                                Delta_Quatr = self.world2quatr(Delta_World)
                                inputs[("translation", i, 0)] = torch.from_numpy(Delta)
                                inputs[("axisangle", i, 0)] =   torch.from_numpy(Delta_Quatr)
                            
                            else:
#                                print('H@H@H@HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
#                                print('Delta_line[0]:' + Delta_line[0])
#                                print('line[0]:' + line[0])
#                                print(self.filenames[index+i])
                                Delta = (np.random.rand(3)-.5)/50 
                                Next_World = self.quatr2world(Ref_Quatr +  (np.random.rand(4)-.5)/50)
                                Delta_World = np.dot(Ref_World, np.linalg.pinv(Next_World))
                                Delta_Quatr = self.world2quatr(Delta_World)
  
                                inputs[("translation", i, 0)] = torch.from_numpy(Delta)
                                inputs[("axisangle", i, 0)] =   torch.from_numpy(Delta_Quatr)   
    
                        else:
                            Delta = (np.random.rand(3)-.5)/50
                            Next_World = self.quatr2world(Ref_Quatr +  (np.random.rand(4)-.5)/50)
                            Delta_World = np.dot(Ref_World, np.linalg.pinv(Next_World))
                            Delta_Quatr = self.world2quatr(Delta_World)
                            
                            inputs[("translation", i, 0)] = torch.from_numpy(Delta)
                            inputs[("axisangle", i, 0)] =   torch.from_numpy(Delta_Quatr)   
                        
                    else:
                        
                        if (index+i >= 0) and (index+i < len(self.filenames)):
                            Delta_line = self.filenames[index+i].split()
    
                            if  (Delta_line[0] == line[0]):
                                #print('Delta')
                                Delta = np.array([np.float(Delta_line[3]), np.float(Delta_line[4]), np.float(Delta_line[5])]) - Ref_Trans
                                Next_Euler = np.array([np.float(Delta_line[6]), np.float(Delta_line[7]), np.float(Delta_line[8])], np.float64)
                                Next_World = self.euler2world(Next_Euler)
                                Delta_World = np.dot(Ref_World, np.linalg.pinv(Next_World))
                                Delta_Euler = self.world2euler(Delta_World)
                                #print(Delta)
                                #inputs[("translation", i, 0)] = torch.from_numpy(np.array([np.float(line[3]), np.float(line[4]), np.float(line[5])]))
                                inputs[("translation", i, 0)] = torch.from_numpy(Delta)
                                inputs[("axisangle", i, 0)] =   torch.from_numpy(Delta_Euler)*57.29577951308232 
                            
                            else:
                                Delta = (np.random.rand(3)-.5)/50 
                                Next_World = self.euler2world(Ref_Euler +  (np.random.rand(3)-.5)/50)
                                Delta_World = np.dot(Ref_World, np.linalg.pinv(Next_World))
                                Delta_Euler = self.world2euler(Delta_World)
                                
                                inputs[("translation", i, 0)] = torch.from_numpy(Delta)
                                inputs[("axisangle", i, 0)] =   torch.from_numpy(Delta_Euler)   
    
                        else:
                            Delta = (np.random.rand(3)-.5)/50
                            Next_World = self.euler2world(Ref_Euler +  (np.random.rand(3)-.5)/50)
                            Delta_World = np.dot(Ref_World, np.linalg.pinv(Next_World))
                            Delta_Euler = self.world2euler(Delta_World)

                            inputs[("translation", i, 0)] = torch.from_numpy(Delta)
                            inputs[("axisangle", i, 0)] =   torch.from_numpy(Delta_Euler)   
    #                        print('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH2')
                        
    
        return inputs






    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
        
        



  
        
        
        
        
        
        
        
        
        
        
        
        
