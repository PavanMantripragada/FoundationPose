import numpy as np
from Utils import *

lock_type = 'dimpled_hard'

test_scene_dir = f'./demo_data/{lock_type}'
gt_poses = np.load(test_scene_dir+"/hole_poses.npy")
pred_poses = np.load(test_scene_dir+"/predicted_hole_poses.npy")
print(compute_orientation_errors(pred_poses, gt_poses[:pred_poses.shape[0]]))
print(compute_position_errors(pred_poses, gt_poses[:pred_poses.shape[0]]))