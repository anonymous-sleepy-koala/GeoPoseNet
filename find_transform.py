

"""Adapted from https://github.com/chrischoy/FCGF/blob/master/demo.py
"""
import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from utils.visualization import get_colored_point_cloud_feature
from utils.misc import extract_features

from models.resunet import GeoPoseNet,ResUNetBN2C
# import utils.open3d_utils as o3d_utils

import torch


def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts



def demo(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load(args.checkpoint)
  model = GeoPoseNet(args.num_points, num_objs=13, D=3)
  model.load_state_dict(checkpoint)
  model.eval()

  model = model.to(device)

  if args.data_type == 'ply':
    pcd_ori = o3d.io.read_point_cloud(args.input)
  elif args.data_type == 'txt':
    xyz = np.loadtxt(args.input)
    pcd_ori = o3d.geometry.PointCloud()
    pcd_ori.points = o3d.utility.Vector3dVector(xyz)
  else:
    pass


  xyz_ori = np.asarray(pcd_ori.points) 
  center_m = np.mean(xyz_ori, axis=0)
  print(center_m)
  # xyz_ori = xyz_ori - center_m
       
  vis_pcd_ori = o3d.geometry.PointCloud()
  vis_pcd_ori.points = o3d.utility.Vector3dVector(xyz_ori)
  vis_pcd_ori.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz_ori) + np.array([[255, 0, 0]]))

  # xyz_occ = np.loadtxt('../data/Linemod_occlusion/ori_models/ape/001.xyz') 
  xyz_occ = np.loadtxt('./ss_01.txt') 
  center_m = np.mean(xyz_occ, axis=0)
  print(center_m)
  # xyz_occ = xyz_occ - center_m
  pcd_occ = o3d.geometry.PointCloud()
  pcd_occ.points = o3d.utility.Vector3dVector(xyz_occ)
  # print(xyz_ori, xyz_occ)

  # pose_pred = o3d.registration.registration_icp(
  #       vis_pcd_ori, pcd_occ, 0.01, np.identity(4),
  #       o3d.registration.TransformationEstimationPointToPoint())

        #             xyz_ori, 
        #             xyz_occ,
        #             args.voxel_size)

  # t1 = np.array([[  1.0000000,  0.0000000,  0.0000000, 0],
  #  [0.0000000, 0,  1, 0],
  #  [0.0000000, -1,  0.0, 0],
  #  [0, 0,0,1]])
  # t2 = np.array([[  0, 1, 0, 0],
  #  [ 1, 0, 0, 0],
  #  [0.0000000, 0,  1, 0],
  #  [0, 0,0,1]])
  # t3 = np.array([[  0, 0, 1, 0],
  #  [ 0, 1, 0, 0],
  #  [-1, 0,  0, 0],
  #  [0, 0,0,1]])

  # trans = np.array([[ 0, -1,  0,  0],
  #                        [ 0,  0,  1,  0],
  #                        [-1,  0,  0,  0],
  #                        [ 0,  0,  0,  1]])
  # trans = t3 @ t1
#   print(trans)
#   trans = np.array([[4.293176663059999898e-01, 9.027915073290000425e-01, -2.624779131250000105e-02, 3.303213316779999853e-01],
# [-8.206503212950000403e-01, 4.020663457079999836e-01, 4.060980329439999870e-01, 6.166886329929999883e-02],
# [3.771657146449999831e-01, -1.528021365370000051e-01, 9.134720905780000511e-01, -9.310759878829999447e-01],
# [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
  # trans = np.linalg.inv(trans) 
  # xyz_ori = apply_transform(xyz_ori, trans)
  vis_pcd_ori.points = o3d.utility.Vector3dVector(xyz_ori)
  

  vis_pcd_occ = o3d.geometry.PointCloud()
  # x = xyz_occ[:, 0][:, None]
  # y = xyz_occ[:, 1][:, None]
  # z = xyz_occ[:, 2][:, None]
  # xyz_occ = np.concatenate((-z, -x, y), axis=1)
  vis_pcd_occ.points = o3d.utility.Vector3dVector(xyz_occ)
  vis_pcd_occ.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz_occ))


  o3d.visualization.draw_geometries([vis_pcd_ori, vis_pcd_occ])



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '-i',
      '--input',
      default='../data/Linemod_preprocessed/models/obj_01.ply',
      type=str,
      help='filename')
  parser.add_argument(
      '--data_type',
      default='ply',
      type=str,
      help='data type')
  parser.add_argument(
      '-c',
      '--checkpoint',
      default='../checkpoints/geopose-fcn-full/val_best_net_M.pth',
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.001,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument(
      '--num_points',
      default=1500,
      type=int,
      help='number of sampled points')

  args = parser.parse_args()
  demo(args)


