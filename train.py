# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function


#from trainer_togetherTrAng import Trainer
#from trainer_together import Trainer
#from trainer_poseOnly_XYZ import Trainer
#from trainer_separate3 import Trainer
#from trainer_poseOnly2 import Trainer
# from trainer_togetherTrAng_Mask import Trainer
from trainer_togTrAng_NoMag_MSK import Trainer
#from trainer_poseOnlyQuat import Trainer
#from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

opts.data_path = '/home/joon/Documents/HPC/Data/' 

#print(opts)
opts.num_workers = 3
opts.dataset='Custom'
opts.split = '3DPrint_Invert'
opts.log_dir = 'WithMag_Inv_MulSclFul_Pre'
opts.png = True
opts.no_eval=True
opts.height=256
opts.width=256
opts.batch_size=8
opts.frame_ids=[0, 2]
opts.num_epochs=40
opts.disparity_smoothness=.01
opts.trans_weight=[1.0, 1.0, 1.0]
opts.use_pose='1'
opts.use_stereo=True
opts.pose_model_input='all'
opts.scales = [0,1,2,3]
opts.load_weights_folder='/home/joon/Documents/Code/Python/MonoDepth2Enodo/WithMag_Inv_MulSclFul/mdp/models/weights_10/'
opts.models_to_load=["encoder", "depth", "pose_encoder", "pose"]
if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
