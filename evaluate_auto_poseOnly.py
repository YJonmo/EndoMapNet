# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import json 
#from evaluate_pose_together import Evaluate
#rom evaluate_pose_custom2 import Evaluate
#from evaluate_pose_custom3 import Evaluate
from evaluate_pose_Only import Evaluate
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

print(opts.load_weights_folder+'/../opt.json')
with open( opts.load_weights_folder+'/../opt.json') as f: 
    data = json.load(f)

opts.pose_model_input=data['pose_model_input']
opts.frame_ids=data['frame_ids'][:-1]
opts.png = data['png']
opts.dataset = data['dataset']
opts.use_stereo = data['use_stereo']
opts.pose_model_input=data['pose_model_input']
opts.num_layers = data['num_layers']
opts.use_pose = data['use_pose']

#opts.models_to_load = data['models_to_load']



if __name__ == "__main__":
    Eval = Evaluate(opts)
    Eval.evaluate()
