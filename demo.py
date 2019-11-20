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

import torch

def demo(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load(args.checkpoint)
  model = GeoPoseNet(args.num_points, num_objs=13, D=3)
  model.load_state_dict(checkpoint)
  model.eval()

  model = model.to(device)

  if args.data_type == 'ply':
    pcd = o3d.io.read_point_cloud(args.input)
  elif args.data_type == 'txt':
    xyz = np.loadtxt(args.input)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
  else:
    pass

  xyz_full, xyz_down, feature = extract_features(
      model,
      np.array(pcd.points),
      voxel_size=args.voxel_size,
      device=device,
      skip_check=True)
  vis_pcd = o3d.geometry.PointCloud()
  vis_pcd.points = o3d.utility.Vector3dVector(xyz_down)

  vis_pcd_full = o3d.geometry.PointCloud()
  xyz_full = np.loadtxt('../data/Linemod_occlusion/ori_models/ape/001.xyz')
  x = xyz_full[:, 0][:, None]
  y = xyz_full[:, 1][:, None]
  z = xyz_full[:, 2][:, None]
  xyz_full = np.concatenate((x, y, z), axis=1)
  vis_pcd_full.points = o3d.utility.Vector3dVector(xyz_full)
  vis_pcd_full.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz_full))

  # feature = np.ones_like(feature.detach().cpu().numpy())
  vis_pcd = get_colored_point_cloud_feature(vis_pcd, 
                                            feature.detach().cpu().numpy(),
                                            args.voxel_size)
  o3d.visualization.draw_geometries([vis_pcd, vis_pcd_full])


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