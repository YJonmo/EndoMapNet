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


if __name__ == "__main__":
    Eval = Evaluate(opts)
    Eval.evaluate()
