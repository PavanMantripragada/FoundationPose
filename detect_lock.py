# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--lock_type', type=str, default=f'pin_tumbler')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh_file = f'{code_dir}/demo_data/{args.lock_type}/mesh/mesh_reconstructed.obj'
  test_scene_dir = f'{code_dir}/demo_data/{args.lock_type}'

  mesh = trimesh.load(mesh_file,process=False)
  # scale_dict = {"pin_tumbler": 0.19, "disc_detainer": 1.0, "dimpled": 1.0, "tubular":1.0}
  # mesh.apply_scale(scale_dict[lock_type])
  # Create coordinate frame (X=red, Y=green, Z=blue)
  frame = trimesh.creation.axis(origin_size=0.001, axis_length=0.0172/2)
  # Combine mesh and frame in a scene
  scene = trimesh.Scene([mesh, frame])
  # Show the scene
  scene.show()
  
  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
  to_origin_old, _ = trimesh.bounds.oriented_bounds(mesh)
  print(to_origin_old)
  # Apply transformation to mesh
  mesh.apply_transform(to_origin_old)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  print(to_origin)
  
  # exit()
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  bbox[0,0] = 0.0
  bbox[1,0] = extents[0]
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  color = reader.get_color(0)
  depth = reader.get_depth(0)
  mask = reader.get_mask(0).astype(bool)
  pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
  center_pose = pose @ np.linalg.inv(to_origin)
  center_pose[:3,:3] = pose[:3,:3]
  center_pose = center_pose @ to_origin_old
  if debug>=1:
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    cv2.imshow('1', vis[...,::-1])
    cv2.waitKey(0)
  np.save(test_scene_dir+"/predicted_hole_pose.npy", center_pose)


