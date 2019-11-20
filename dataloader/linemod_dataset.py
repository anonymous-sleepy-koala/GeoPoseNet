#!/usr/bin/python

import os
import numpy as np
import time
import random
import yaml
from yaml import CLoader as Loader # way faster 
import open3d as o3d
from dataloader.base_dataset import BaseDataloader
from dataloader.object_model import ObjectModel
from dataloader.camera_model import CameraModel
from dataloader.data_utils import *
import utils.tf_numpy as tf_numpy
import dataloader.data_utils as dutils
import MinkowskiEngine as ME
import MinkowskiEngine.utils

import matplotlib.pyplot as plt

class LineMODObjectModel(ObjectModel):
    def __init__(self, root_dir, idx, index, name, is_sym, num_points):
        super(LineMODObjectModel, self).__init__(root_dir, idx, index, name, num_points)

        self.is_sym = is_sym
        self.diameter = self._read_diameter()

    def _read_ply_file(self, filename):
        f = open(filename)
        assert f.readline().strip() == "ply"
        f.readline()
        f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def _read_model_pts(self):
        '''
        get Polygon data of the model
        '''
        item = '%02d' % self.id
        self.model_path = os.path.join(self.root_dir, 'models')
        ply_file = os.path.join(self.model_path, 'obj_'+item+'.ply')
        model_points = self._read_ply_file(ply_file)
        
        return model_points

    def _read_diameter(self):
        '''
        get model diameter
        '''
        info_file = os.path.join(self.root_dir, 'models', 'models_info.yml')
        model_info = yaml.load(open(info_file, 'r'), Loader=Loader)
        diameter = model_info[self.id]['diameter'] / 1000.0 # in meter

        return diameter

class LineMOD(BaseDataloader):
    def __init__(self, root_dir, split, voxel_size, num_points=500, 
        img_height=480, img_width=640, select_obj=None):
        super(LineMOD, self).__init__(split, root_dir, img_height, img_width)

        self.voxel_size = voxel_size
        self.select_obj = select_obj
        self.obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.obj_dics = {
            1:  'ape',
            2:  'benchwise',
            4:  'camera',
            5:  'can',
            6:  'cat',
            8:  'driller',
            9:  'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            13: 'iron',
            14: 'lamp',
            15: 'phone'
        }

        self.paths, self.list_class, self.list_inclass_index = self.get_paths()
        self.meta = self.get_meta()
        
        self.num_points = num_points
        self.symmetric_obj_idx = [7, 8] # index, 0, 1, 2, ...
        
        self.models = self.get_models()

        # camera parameters
        self.cam_model = CameraModel(cam_cx=325.26110, cam_cy=242.04899, 
            cam_fx=572.41140, cam_fy=573.57043)
        print('found {} images for the {} dataset'.format(
            self.__len__(), self.split))

    def get_models(self):
        '''
        collect model informaiton
        '''
        models = {}
        for idx in range(len(self.obj_ids)):
            obj_index = idx # continuous: 0, 1, 2, ...
            obj_id = self.obj_ids[idx] # discontinuous: 1, 2, 4, ...
            is_sym = self.is_symmetric_obj(obj_index)
            model = LineMODObjectModel(self.root_dir, obj_id, obj_index, self.obj_dics[obj_id], is_sym, self.num_points)
            model.load_model(self.root_dir)
            models[obj_id] = model
        return models

    def get_paths(self):
        '''
        get data path of LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_posecnn_mask = []

        list_class = []
        list_inclass_index = [] 

        for index in self.obj_ids:
            if self.select_obj is not None:
                if not index in self.select_obj:
                    continue
            item = '%02d' % index
            cur_dir = os.path.join(self.root_dir, 'data', item)
            
            gt_file = os.path.join(cur_dir, 'gt.yml') # ground truth meta file: pose, bbox
            list_file = open( os.path.join(self.root_dir, 'data', \
                item, self.split+'.txt') )

            line = list_file.readline().strip()
            #temp_num = 0
            while line:
                rgb_file = os.path.join(cur_dir, 'rgb', line + '.png')
                depth_file = os.path.join(cur_dir, 'depth', line + '.png')
                if self.split == 'test':
                    mask_file = os.path.join(self.root_dir, 'segnet_results', item+'_label', line+'_label.png')
                    # mask_file = os.path.join('../checkpoints/temp/images_val', item, line + '.png')
                else:
                    mask_file = os.path.join(cur_dir, 'mask', line + '.png')

                posecnn_mask = os.path.join(self.root_dir, 'segnet_results', item+'_label', line+'_label.png')

                paths_rgb.append(rgb_file)
                paths_depth.append(depth_file)
                paths_mask.append(mask_file)
                paths_posecnn_mask.append(posecnn_mask)

                list_class.append(index) # class index of each item
                list_inclass_index.append(int(line)) # item index within corresponding class

                line = list_file.readline().strip()
                #temp_num += 1
            #print('item:', item, 'num', temp_num)

        assert len(paths_rgb)>0, "rgb images not found."

        paths = {
           "rgb" : paths_rgb,
           "depth" : paths_depth,
           "mask" : paths_mask,
           'posecnn_mask': paths_posecnn_mask
           }

        return paths, list_class, list_inclass_index

    def get_meta(self):
        '''
        get meta data: rotation, translation, bbox
        '''
        meta = {}
        for index in self.obj_ids:
            item = '%02d' % index
            cur_dir = os.path.join(self.root_dir, 'data', item)
            meta_file = os.path.join(cur_dir, 'gt.yml')

            if index == 2:
                meta_raw = yaml.load(open(meta_file), Loader=Loader)
                meta_extracted = {}
                for i in range(0, len(meta_raw)):
                    for j in range(len(meta_raw[i])):
                        if meta_raw[i][j]['obj_id'] == 2:
                            meta_extracted[i] = [meta_raw[i][j]]
                            break
                meta[index] = meta_extracted
            else:
                meta[index] = yaml.load(open(meta_file), Loader=Loader)

        return meta

    def preprocess_mask(self, mask, depth):
        mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
        if self.split == 'test': # use results from segnet
            mask_label = np.ma.getmaskarray(np.ma.masked_equal(mask, np.array(255)))
        else:
            mask_label = np.ma.getmaskarray(np.ma.masked_equal(mask, np.array([255, 255, 255])))[:, :, 0]
        
        res = mask_label * mask_depth
        if np.count_nonzero(res) == 0:
            res = np.ones_like(res)
        return res

    def get_mask_bbox(self, mask):
        '''return the bounding box containing mask
        '''
        index = np.nonzero(mask)
        try:
            rmin = min(index[0]) #row
        except:
            print(mask)
        rmax = max(index[0])
        cmin = min(index[1]) #col
        cmax = max(index[1])
        # return [cmin, rmin, cmax-cmin+1, rmax-rmin+1]
        return [rmin, rmax, cmin, cmax]

    def get_mask_inside_bbox(self, mask, bbox_index):
        mask_in_bbox = mask[bbox_index].flatten().nonzero()[0]
        if len(mask_in_bbox) == 0:
            cc = 0
            return (cc, cc, cc, cc, cc, cc)

        if len(mask_in_bbox) > self.num_points:
            c_mask = np.zeros(len(mask_in_bbox), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            mask_in_bbox = mask_in_bbox[c_mask.nonzero()]
        else:
            pass#mask_in_bbox = np.pad(mask_in_bbox, (0, self.num_points - len(mask_in_bbox)), 'wrap')

        return mask_in_bbox

    
    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        depth = depth_read(self.paths['depth'][index])
        mask_raw = mask_read(self.paths['mask'][index])
        mask = self.preprocess_mask(mask_raw, depth)
        if not np.any(mask):
            print(self.paths['mask'][index])
            mask = np.invert(mask)
            print(mask)
        #mask_posecnn = seg_mask_read(self.paths['posecnn_mask'][index])

        obj = self.list_class[index]
        inclass_index = self.list_inclass_index[index]
        meta = self.meta[obj][inclass_index][0]

        trans = np.array(meta['cam_t_m2c']) / 1000.0 # in meter
        rot = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        pose = np.concatenate((rot, trans[np.newaxis].T), 1) # 3 x 4 matrix
        # bbox = get_bbox_linemod(self.get_mask_bbox(mask))
        bbox = self.get_mask_bbox(mask)

        seg_mask = seg_mask_read(self.paths['mask'][index])

        candidates = {
           'rgb': rgb,
           'depth': depth,
           'mask': mask,
           'seg_mask': seg_mask,
           #'posecnn_mask': mask_posecnn,
           'bbox': bbox,
           'pose': pose,
           'model': self.models[obj],
        }
        return candidates
   
    # def create_fpfh_feature(self, pts):
    #     pcd = o3d.PointCloud()
    #     pcd.points = o3d.Vector3dVector(pts)
    #     estimate_normals(pcd,
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=60))
    #     fpfh = o3d.registration.compute_fpfh_feature(pcd,
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=60))
    #     # estimate_normals(pcd,
    #     #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    #     # fpfh = o3d.registration.compute_fpfh_feature(pcd,
    #     #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    #     return fpfh.data
    # self.corners = np.array([
    #         [min_x, min_y, min_z],
    #         [min_x, min_y, max_z],
    #         [min_x, max_y, min_z],
    #         [min_x, max_y, max_z],
    #         [max_x, min_y, min_z],
    #         [max_x, min_y, max_z],
    #         [max_x, max_y, min_z],
    #         [max_x, max_y, max_z],
    #     ])

    def draw_reprojection(self, index, im, pts, pose, name):
        
        pts_i = self.model2img(pts, pose)

        img = plt.imshow(im)
        plt.scatter(x=pts_i[:, 0], y=pts_i[:, 1], c='r', s=.1)
        # implot = plt.imshow(img)
        # put a red dot, size 40, at 2 locations:
        loc = '/home/mylin/Dropbox (MIT)/paper_cvpr2020/point_cloud/'
        plt.savefig(loc + name + '_' + str(index)+'.eps')
        plt.close()
        # plt.show()

    def draw_bbox(self, index, im, pts, pose, pose_pred):
        pt_list = np.array([[0, 1], [0,2],[0, 4], [1,3], [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]])
        pts_i = self.model2img(pts, pose)
        for i in range(12):
            p1 = pts_i[pt_list[i, 0], :]
            p2 = pts_i[pt_list[i, 1], :]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', lw=1)

        pts_i = self.model2img(pts, pose_pred)
        for i in range(12):
            p1 = pts_i[pt_list[i, 0], :]
            p2 = pts_i[pt_list[i, 1], :]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=1)

        img = plt.imshow(im)
        # plt.scatter(x=pts_i[:, 0], y=pts_i[:, 1], c='r', s=1)
        # implot = plt.imshow(img)
        # put a red dot, size 40, at 2 locations:
        
        loc = '/home/mylin/Dropbox (MIT)/paper_cvpr2020/pose_projection_figure/'
        plt.savefig(loc  + str(index)+'.eps')
        plt.close()