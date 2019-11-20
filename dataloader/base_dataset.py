import os.path
import numpy as np
import random
import yaml
from abc import abstractmethod
import open3d as o3d
import torch
import torch.utils.data as data
import MinkowskiEngine as ME
import MinkowskiEngine.utils

from dataloader.transforms import image_transforms
# from dataloader.data_utils import *
from utils.metrics import AverageMeter

import time



class BaseDataloader(data.Dataset):
    """
    A base data loader
    """
    def __init__(self, split, root_dir, img_height=480, img_width=640,
        do_augmentation=True):
        '''
        args:
            root_dir: root directory of dataset
            split: 'train' | 'test'

        '''
        self.split = split
        self.root_dir = root_dir

        self.symmetric_obj_idx = []

        self.do_augmentation = do_augmentation
        self.transforms = image_transforms(
            mode=split, 
            do_augmentation=do_augmentation)

        self.img_height = img_height
        self.img_width = img_width
        self.ymap = np.array([[j for i in range(self.img_width)] \
            for j in range(self.img_height)])
        self.xmap = np.array([[i for i in range(self.img_width)] \
            for j in range(self.img_height)])
        self.timer = AverageMeter()

    def get_models(self):
        '''
        collect object model informaiton
        '''
        raise NotImplementedError()

    def get_num_of_models(self):
        return len(self.obj_ids)

    def get_name_of_models(self, model_index):
        return self.obj_dics[self.obj_ids[model_index]]

    def is_symmetric_obj(self, model_index):
        return model_index in self.symmetric_obj_idx

    def model2cam(self, pts_m, pose):
        '''tranform points to camera coordinate system
        args:
            pts_m: Nx3 point array, coordiantes in model frame
            pose: 3x4 transformation matrix, from model to cam
        '''
        return np.matmul(pts_m, pose[:3, :3].T) + pose[:3, 3:].T

    def cam2img(self, pts_c):
        '''project points in camera frame to image
        args:
            pts_c: Nx3 array, coordiantes in camera frame
        return:
            pts_i: Nx2 array, projected pixel coordinates
        '''
        pts_i = np.matmul(pts_c, self.matrix_K.T)
        pts_i = pts_i[:, :2] / pts_i[:, 2:]
        return pts_i

    def model2img(self, pts_m, pose):
        return self.cam2img(self.model2cam(pts_m, pose))

    def get_pts_sensor_cam(self, depth, bbox_index, mask_in_bbox):
        '''
        recover point cloud of object from RGB image and depth map
        args:
            index: index of object in the image

        '''
        depth_masked = depth[bbox_index].flatten()[mask_in_bbox][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[bbox_index].flatten()[mask_in_bbox][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[bbox_index].flatten()[mask_in_bbox][:, np.newaxis].astype(np.float32)

        pts_sensor_cam = self.cam_model.img2cam(depth_masked, xmap_masked, ymap_masked) 
        return pts_sensor_cam

    def get_gt_vector_field(self, pts_farthest, pts_obs):
        '''for each points of observed point cloud, find the direction 
             vector from the point to farthest points
             works for both 3D and 2D

        args: 
            pts_farthest_c: mx3 (or mx2) array
            pts_obs_c: nx3 (or nx2) array
        '''
        # print('input farthest', pts_farthest.shape)
        # print('input obs', pts_obs.shape)
        num_k = pts_farthest.shape[0]
        num_pts = pts_obs.shape[0]
        pts_obs = np.repeat(pts_obs[:, None, :], num_k, axis=1)
        pts_farthest = np.repeat(pts_farthest[None, :, :], num_pts, axis=0)
        
        # print('processed farthest', pts_farthest.shape)
        # print('processed obs', pts_obs.shape)
        dir_vecs = pts_farthest - pts_obs

        norm = np.linalg.norm(dir_vecs, axis=2, keepdims=True)
        norm[norm<1e-3] += 1e-3
        dir_vecs /= norm

        return dir_vecs

    def draw_vector_field_2D(self, pts, vectors):
        pass

    def draw_vector_field_3D(self, pts, vectors):
        N = pts.shape[0]
        line_set = o3d.geometry.LineSet()
        end_pt = pts + vectors
        line_set.points = o3d.utility.Vector3dVector(np.concatenate((pts, end_pt), axis=0))
        lines = [ [i, i+N] for i in range(N)]

        line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set])

        # pcd = o3d.PointCloud()
        # pcd.points = o3d.Vector3dVector(pts)
        # # pcd.colors = o3d.Vector3dVector(vis_colors)
        # # pcd = self.model_points
        # o3d.draw_geometries([pcd]) 

    @abstractmethod
    def preprocess(self, raws):
        raise NotImplementedError()

    def __len__(self):
        return len(self.paths['rgb']) 

    @abstractmethod
    def __getraw__(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        start = time.time()
        raws = self.__getraw__(index)
        preprocessed_data = self.preprocess(index, raws)
        # candidates = self.transforms(preprocessed_data)
        end = time.time() 
        self.timer.update(end-start)
        # items = {key:val for key, val in candidates.items() if val is not None}
        return preprocessed_data

    def preprocess(self, index, raws):
        '''extra processing after calling __getraw__
        '''
        # if index > 0:
        #     return

        pose_gt = np.concatenate((raws['pose'], np.array([[0,0,0,1]]) ), 0)
        # print(pose_gt)
        # pose_gt = raws['pose']
        ymin, ymax, xmin, xmax = raws['bbox']
        # bbox_index = np.s_[ymin:ymax+1, xmin:xmax+1]
        bbox_index = np.s_[ymin:ymax, xmin:xmax]
        mask_in_bbox = self.get_mask_inside_bbox(raws['mask'], bbox_index)

        depth_masked = raws['depth'][bbox_index]
        # print(bbox_index)
        pts_sensor_cam = self.get_pts_sensor_cam(raws['depth'], bbox_index, mask_in_bbox)
        model_points = raws['model'].get_model_points()

        model_index = raws['model'].get_index()

        
        xyz_m = model_points#dutils.sample_knn(model_points, int(self.num_points*0.6), ref_pt='other') #model_points
        xyz_s = pts_sensor_cam#raws['model']._downsample_points(1000)#model_points 

        # mm = apply_transform(xyz_m, pose_gt)
        # np.savetxt('mm_' + str(index) + str(self.select_obj[0]) + '.txt', mm)
        # np.savetxt('ss_' + str(index) + str(self.select_obj[0]) + '.txt', xyz_s)
        if True:#self.split == 'train':#self.random_rotation:
            #self.randg = np.random.RandomState()
            #self.rotation_range = 180
            #T = dutils.sample_random_trans(xyz_m, self.randg, self.rotation_range)
            #trans = T  
            #xyz_m = dutils.apply_transform(xyz_m, T)
            pass
            # gt_dir_s = self.apply_transform(gt_dir_s, T)
        else:
            #trans = pose_gt
            pose_gt = np.identity(4)

        pts_model_cam = self.model2cam(model_points, pose_gt)
        pts_farthest = raws['model'].get_farthest_points()
        pts_farthest_cam = self.model2cam(pts_farthest, pose_gt)
        # pts_farthest_i = self.model2img(pts_farthest, raws['pose'])
        gt_dir_s = self.get_gt_vector_field(pts_farthest_cam, xyz_s)
        # rgb_selected = rgb_masked.flatten()[mask_in_bbox][:, np.newaxis].astype(np.float32)
        # gt_dir_vecs_i = self.get_gt_vector_field(pts_farthest_i, rgb_selected)
        
        # self.draw_vector_field_3D(pts_sensor_cam, gt_dir_vecs_c[:, 0, :])

        center_m = np.mean(xyz_m, axis=0)
        center_s = np.mean(xyz_s, axis=0)

        xyz_m = xyz_m - center_m
        xyz_s = xyz_s - center_s

        # Voxelization
        sel_m = ME.utils.sparse_quantize(xyz_m / self.voxel_size, return_index=True)
        sel_s = ME.utils.sparse_quantize(xyz_s / self.voxel_size, return_index=True)
        #print('m', xyz_m.shape, 's', xyz_s.shape)
        xyz_m = xyz_m[sel_m]
        xyz_s = xyz_s[sel_s]
        #print('downsample m', xyz_m.shape, 's', xyz_s.shape)
        
        coords_m = np.floor(xyz_m / self.voxel_size)
        coords_s = np.floor(xyz_s / self.voxel_size)

        coords_m, indices_m = np.unique(coords_m, axis=0, return_index=True)
        coords_s, indices_s = np.unique(coords_s, axis=0, return_index=True)
        
        xyz_m = xyz_m[indices_m]
        xyz_s = xyz_s[indices_s]
        
        # Get features
        npts_m = len(xyz_m)
        npts_s = len(xyz_s)

        feats_train_m = []
        feats_train_s = []

        feats_train_m.append(np.ones((npts_m, 1)))
        feats_train_s.append(np.ones((npts_s, 1)))

        feats_m = np.hstack(feats_train_m)
        feats_s = np.hstack(feats_train_s)
        
        # Get coords
        gt_dir_s = gt_dir_s[sel_s][indices_s]

        if False:#self.transform:
          coords_m, feats_m = self.transform(coords_m, feats_m)
          coords_s, feats_s = self.transform(coords_s, feats_s)

        #print(xyz_m.shape, gt_dir_s.shape)
        xyz_m += center_m
        xyz_s += center_s
        # print(xyz_m, coords_m, feats_m, center_m, 
        #         xyz_s, coords_s, feats_s, center_s,
        #         raws['model'], pts_farthest, pts_farthest_cam, gt_dir_s, pose_gt)
        
        #a = raws['model']._downsample_points(200)#model_points 

        #if index % 250 == 0:
        #    pose_pred = np.loadtxt('../pose_results/'+str(index)+'.txt')
        #    self.draw_reprojection(index, raws['rgb'], a, pose_gt[:3, :], 'gt')
        #    self.draw_reprojection(index, raws['rgb'], a, pose_pred[:3], 'pred')

        #a = raws['model'].get_corner_points()#model_points 

        #if index % 250 == 0:
        #    pose_pred = np.loadtxt('../pose_results/'+str(index)+'.txt')
        #    self.draw_bbox(index, raws['rgb'], a, pose_gt[:3, :], pose_pred[:3, :])
        #    # self.draw_bbox(index, raws['rgb'], a, pose_pred[:3], 'pred')

        return (xyz_m, coords_m, feats_m, center_m, 
                xyz_s, coords_s, feats_s, center_s,
                raws['model'], pts_farthest, pts_farthest_cam, gt_dir_s, pose_gt)
        
