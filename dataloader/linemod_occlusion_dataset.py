#!/usr/bin/python

import os
import numpy as np
import random
import time
import yaml
from yaml import CLoader as Loader # way faster 
import open3d as o3d

from dataloader.base_dataset import BaseDataloader
from dataloader.object_model import ObjectModel
from dataloader.camera_model import CameraModel
from dataloader.data_utils import *



#from open3d.open3d.geometry import estimate_normals
def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

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

class LineMODOcclusion(BaseDataloader):
    def __init__(self, root_dir, 
        split, voxel_size, num_points=500, img_height=480, img_width=640, select_obj=None):
        super(LineMODOcclusion, self).__init__(split, root_dir, img_height, img_width)

        self.voxel_size = voxel_size
        self.obj_ids = [1, 5, 6, 8, 9, 10, 11, 12] # ignore 11: benchvise
        self.select_obj = select_obj
        self.obj_dics = {
            1:  'ape',
            2: 'benchvise',
            5:  'can',
            6:  'cat',
            8:  'driller',
            9:  'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
        }

        self.paths, self.list_class, self.list_inclass_index = self.get_paths()
        
        self.num_points = num_points
        self.symmetric_obj_idx = [7, 8]
        self.depth_factor = 1000.0 # the unit of depth image is meter
        self.models = self.get_models()

        # camera projection model
        self.cam_model = CameraModel(cam_cx=325.26110, cam_cy=242.04899, 
            cam_fx=572.41140, cam_fy=573.57043)
        
        print('found {} images for the {} dataset'.format(
            self.__len__(), self.split))

    def get_paths(self):
        '''
        get data path of LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_pose = []

        list_class = []
        list_inclass_index = [] 

        for index in self.obj_ids:
            if self.select_obj is not None:
                if not index in self.select_obj:
                    continue
            obj_name = self.obj_dics[index]
            item = '%02d' % index
            cur_dir = os.path.join(self.root_dir, 'data', item)
            gt_file = os.path.join(cur_dir, 'gt.yml') # ground truth meta file: pose, bbox

            list_file = open(os.path.join(self.root_dir, self.obj_dics[index]+'_val.txt'))
            line = list_file.readline().strip()
            while line:
                file_index = line.split('/')[2].split('_')[1].split('.')[0]
                
                rgb_file = os.path.join(self.root_dir, line)
                depth_file = os.path.join(self.root_dir, 'RGB-D', 'depth_noseg', 'depth_'+file_index+'.png')
                pose_file = os.path.join(self.root_dir, 'poses', obj_name, 'info_'+file_index+'.txt')
                if False:#self.split == 'test':
                    mask_file = os.path.join(self.root_dir, 'segnet_results', item+'_label', line+'_label.png')
                else:
                    mask_file = os.path.join(self.root_dir, 'masks', 
                        obj_name, str(int(file_index)) + '.png')
                # print(rgb_file, mask_file, depth_file, pose_file)
                paths_rgb.append(rgb_file)
                paths_depth.append(depth_file)
                paths_mask.append(mask_file)
                paths_pose.append(pose_file)

                list_class.append(index) # class index of each item
                list_inclass_index.append(int(file_index)) # item index within corresponding class

                line = list_file.readline().strip()

        assert len(paths_rgb)>0, "rgb images not found."

        paths = {
            "rgb" : paths_rgb,
            "depth" : paths_depth,
            "pose" : paths_pose,
            "mask" : paths_mask,
            }

        return paths, list_class, list_inclass_index

    def get_models(self):
        '''
        collect model informaiton
        '''
        models = {}
        for idx in range(len(self.obj_ids)):
            obj_index = idx # continuous: 0, 1, 2, ...
            obj_id = self.obj_ids[idx] # discontinuous: 1, 2, 4, ...
            obj_name = self.obj_dics[obj_id]
            is_sym = self.is_symmetric_obj(obj_index)
            model = LineMODObjectModel(self.root_dir, obj_id, obj_index, obj_name, is_sym, self.num_points)
            model.load_model(self.root_dir)
            models[obj_id] = model
        return models

    @staticmethod
    def read_gt_pose(pose_path):
        with open(pose_path) as pose_info:
            lines = [line[:-1] for line in pose_info.readlines()]
            if 'rotation:' not in lines:
                return np.array([])
            row = lines.index('rotation:') + 1
            rotation = np.loadtxt(lines[row:row + 3])
            translation = np.loadtxt(lines[row + 4:row + 5])
        return np.concatenate([rotation, np.reshape(translation, newshape=[3, 1])], axis=-1)

    def preprocess_mask(self, mask, depth):
        num_thresh = 100
        mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
        mask_label = np.ma.getmaskarray(np.ma.masked_equal(mask, np.array(1)))
        res = mask_label * mask_depth
        if np.count_nonzero(res) < num_thresh:
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

    @staticmethod
    def create_fpfh_feature(self, pts):
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(pts)
        estimate_normals(pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=60))
        fpfh = o3d.registration.compute_fpfh_feature(pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=60))
        # estimate_normals(pcd,
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        # fpfh = o3d.registration.compute_fpfh_feature(pcd,
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        return fpfh.data
    
    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        depth = depth_read(self.paths['depth'][index], 
            depth_factor=self.depth_factor)
        mask_raw = mask_read(self.paths['mask'][index])
        mask = self.preprocess_mask(mask_raw, depth)
        # print(np.count_nonzero(mask))
        if not np.any(mask):
            print(self.paths['mask'][index])
            mask = np.invert(mask)
            print(mask)

        pose = self.read_gt_pose(self.paths['pose'][index])
        # T0 = np.array([[ 0, -1,  0,  0],
        #                  [ 0,  0,  1,  0],
        #                  [-1,  0,  0,  0],
        #                  [ 0,  0,  0,  1]])
        # T1 = np.concatenate((pose, np.array([[0,0,0,1]]) ), 0)
        # T1 = np.linalg.inv(T1) 
        # pose = T1
        # pose = pose[:3, :]
        pose[2, 3] = -pose[2, 3]
        
        obj_id = self.list_class[index]
        bbox = self.get_mask_bbox(mask)

        seg_mask = seg_mask_read(self.paths['mask'][index])

        if np.count_nonzero(mask) == 0:
            return print('error')

        candidates = {
           'rgb': rgb,
           'depth': depth,
           'mask': mask,
           'seg_mask': seg_mask,
           'bbox': bbox,
           'pose': pose,
           'model': self.models[obj_id],
        }
        return candidates
   

    def draw_reprojection(image, pts, pose):
        pts_c = self.model2cam(pts, pose)
        pts_i = self.model2img(pts_c, pose)
        im = plt.imread(image_name)
        implot = plt.imshow(im)
        # put a red dot, size 40, at 2 locations:
        plt.scatter(x=pts_i[:, 0], y=pts_i[i:, 1], c='r', s=40)
        plt.show()
        pls.save('ss.png')


     