import os
import numpy as np
import shutil

dir_path = "demo_data/tubular_hard"
rgb_path = dir_path + "/rgb"
depth_path = dir_path + "/depth"

timestamps = np.load(dir_path + "/timestamps.npy")

depth_selected_dir = os.path.join(dir_path, "depth_selected")
rgb_selected_dir = os.path.join(dir_path, "rgb_selected")

# Create the directories if they don't exist
os.makedirs(depth_selected_dir, exist_ok=True)
os.makedirs(rgb_selected_dir, exist_ok=True)

for timestamp in timestamps:
    depth_image_path = os.path.join(depth_path, str(timestamp) + ".png")
    rgb_image_path = os.path.join(rgb_path, str(timestamp) + ".png")
    
    if os.path.exists(depth_image_path) and os.path.exists(rgb_image_path):
        # Copy the depth image to the depth_selected directory
        shutil.copy(depth_image_path, depth_selected_dir)
        
        # Copy the RGB image to the rgb_selected directory
        shutil.copy(rgb_image_path, rgb_selected_dir)