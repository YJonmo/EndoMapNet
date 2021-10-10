# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

#from evaluate_pose_together import Evaluate
#rom evaluate_pose_custom2 import Evaluate
#from evaluate_pose_custom3 import Evaluate
from evaluate_pose_Only import Evaluate
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()



opts.data_path = '/home/joon/Documents/HPC/Data/' 

#print(opts)
opts.num_workers = 1 
opts.dataset='Custom'
opts.split = '2020-09-18-14-03-36'
#opts.log_dir = 'Test'
opts.png = True
#opts.no_eval=True
opts.height=256
opts.width=256
opts.batch_size=1
opts.frame_ids=[0, 2]
#opts.pose_model_input= 'pairs'
#opts.num_epochs=60
opts.disparity_smoothness=.01
#opts.trans_weight=[.5, .5, 1.0]
opts.use_pose='1'
opts.use_stereo=True
opts.pose_model_input='all'
opts.scales = [0]
opts.load_weights_folder='/home/joon/Documents/Code/Python/MonoDepth2Enodo/WithMag_Inv_Pre/mdp/models/weights_4'
opts.models_to_load=['pose_encoder', 'pose']



if __name__ == "__main__":
    Eval = Evaluate(opts)
    Eval.evaluate()
