import numpy as np
import os

# The recordings of objects with continous videos have
# initial frames without any object in the scene.
# This script removes those frames from the depth and rgb folders.

dir_path = "demo_data/tubular"

depth_path = dir_path + "/depth"
rgb_path = dir_path + "/rgb"

try:
    timestamps = np.load(dir_path+"/timestamps.npy")
except FileNotFoundError:
    print("No timestamps.npy file found. This code should be used only for timestamped image sequences from a video recording of the object.")
    exit()

first_timestamp = timestamps[0]

for filename in os.listdir(depth_path):
    if filename.endswith(".png"):
        image_timestamp = int(filename.split(".")[0])
        if image_timestamp < first_timestamp:
            os.remove(os.path.join(depth_path, filename))

for filename in os.listdir(rgb_path):
    if filename.endswith(".png"):
        image_timestamp = int(filename.split(".")[0])
        if image_timestamp < first_timestamp:
            os.remove(os.path.join(rgb_path, filename))