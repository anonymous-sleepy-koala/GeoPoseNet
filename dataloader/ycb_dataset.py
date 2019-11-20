#!/usr/bin/python

import os
import numpy as np
import random
import yaml
from yaml import CLoader as Loader # way faster 
import scipy.io as scio
import torchvision.transforms as transforms

from dataloader.base_dataset import *
from dataloader.data_utils import *



class YCBObjectModel(ObjectModel):
    def __init__(self, root_dir, idx, index, name, num_points):
        '''
        args:
            name: e.g. '002_master_chef_can', '003_cracker_box'
        '''
        super(YCBObjectModel, self).__init__(root_dir, idx, index, name, num_points)
        self.diameter = 0.02

    def _read_model_pts(self):
        self.model_path = os.path.join(self.root_dir, 'models')
        input_file = open(os.path.join(self.model_path, self.name, 'points.xyz'))
        pts = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            pts.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        
        return np.array(pts)

class YCBVideo(BaseDataloader):
    def __init__(self, root_dir, split, add_noise, num_points=1000, img_height=480, img_width=640):
        '''
        args:
            root_dir: root directory of dataset
            split: 'train' | 'test'

        '''
        super(YCBVideo, self).__init__(split, root_dir, img_height, img_width)

        self.obj_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, \
            19, 21, 24, 25, 35, 36, 37, 40, 51, 52, 61]
        self.obj_dics = {}
        self.get_classes()

        self.list = [] # list of data index
        self.paths = self.get_paths()
        
        self.num_points = num_points
        self.minimum_num_pts = 50
        self.symmetric_obj_idx = [12, 15, 18, 19, 20]
        
        self.models = self.get_models()

        self.add_noise = add_noise 
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        # camera parameters
        self.cam_cx = 312.9869
        self.cam_cy = 241.3109
        self.cam_fx = 1066.778
        self.cam_fy = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.matrix_K = np.array([[self.cam_fx,              0., self.cam_cx],
                                  [             0., self.cam_fy,  self.cam_cy],
                                  [             0.,             0.,            1.]])

        self.matrix_K2 = np.array([[self.cam_fx_2,              0.,  self.cam_cx_2],
                                  [             0.,  self.cam_fy_2,  self.cam_cy_2],
                                  [             0.,             0.,              1.]])
        print('found {} images for the {} dataset'.format(
            self.__len__(), self.split))

    def get_classes(self):
        '''Read object class name from file
        '''
        class_file = open('./dataloader/config/ycb/classes.txt')
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_id = int( class_input.split('_')[0] )
            self.obj_dics[class_id] = class_input.rstrip('\n')

    def get_models(self):
        '''
        collect model informaiton
        '''
        models = {}
        for idx in range(len(self.obj_ids)):
            obj_index = idx # continuous: 0, 1, 2, ...
            obj_id = self.obj_ids[idx] # discontinuous: 1, 2, 4, ...
            model = YCBObjectModel(self.root_dir, obj_id, obj_index, 
                self.obj_dics[obj_id], self.num_points)
            model.load_model(self.root_dir, unit_scale=1.0) # no need to convert to meter
            models[obj_id] = model
        return models

    def get_paths(self):
        '''
        get data path of LineMOD dataset
        '''
        if self.split == 'train':
            list_path = 'dataloader/config/ycb/train_data_list.txt'
        elif self.split == 'test':
            list_path = 'dataloader/config/ycb/test_data_list.txt'
        self.list = []
        self.real = []
        self.syn = []
        input_file = open(list_path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_meta = []

        for index in self.list:
            rgb_file = os.path.join(self.root_dir, index+'-color.png')
            depth_file = os.path.join(self.root_dir, index+'-depth.png')
            mask_file = os.path.join(self.root_dir, index+'-label.png')
            meta_file = os.path.join(self.root_dir, index+'-meta.mat')

            paths_rgb.append(rgb_file)
            paths_depth.append(depth_file)
            paths_mask.append(mask_file)
            paths_meta.append(meta_file)

        assert len(paths_rgb)>0, "rgb images not found."

        paths = {
            "rgb" : paths_rgb,
            "depth" : paths_depth,
            "mask" : paths_mask,
            "meta": paths_meta
            }

        return paths

    def preprocess_mask(self, mask_raw, depth, obj):
        mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
        iter_num = 0
        while 1:
            assert np.any(mask_raw) == True
            iter_num += 1
            idx = np.random.randint(0, len(obj))
            mask_label = np.ma.getmaskarray(np.ma.masked_equal(mask_raw, obj[idx]))
            mask = mask_label * mask_depth

            if iter_num >= 10*len(obj):
                print('infinite loop')
            if len(mask.nonzero()[0]) > self.minimum_num_pts:
                break
        
        return mask, mask_label, idx

    def get_mask_inside_bbox(self, mask, bbox_index):
        mask_in_bbox = mask[bbox_index].flatten().nonzero()[0]

        if len(mask_in_bbox) > self.num_points:
            c_mask = np.zeros(len(mask_in_bbox), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            mask_in_bbox = mask_in_bbox[c_mask.nonzero()]
        else:
            mask_in_bbox = np.pad(mask_in_bbox, (0, self.num_points - len(mask_in_bbox)), 'wrap')
        return mask_in_bbox

    def add_front_image(self, label):
        self.front_num = 2
        front = None
        mask_front = None
        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front_image_file = os.path.join(self.root_dir, seed+'-color.png')
                front_image = Image.open(front_image_file).convert("RGB")
                front = np.array(self.trancolor(front_image))
                # front = np.transpose(front, (2, 0, 1))
                f_label_file = os.path.join(self.root_dir, seed+'-label.png')
                f_label = np.array(Image.open(f_label_file))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = np.ma.getmaskarray(np.ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break
        return front, mask_front, add_front, label

    def _to_dataloader_index(self, dataset_id):
        '''in the labeling of dataset, index of each object: 1, 2, ..., 21
             in the dataloader, index of each object: 0, 1, ..., 20
                                id of each object: 2, 3, ..., 61
        '''
        return dataset_id - 1

    def _to_dataloader_id(self, dataset_id):
        '''in the labeling of dataset, index of each object: 1, 2, ..., 21
             in the dataloader, index of each object: 0, 1, ..., 20
                                id of each object: 2, 3, ..., 61
        '''
        index = self._to_dataloader_index(dataset_id)

        return self.obj_ids[index]

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        mask_raw = mask_read(self.paths['mask'][index])
        

        mask_back = np.ma.getmaskarray(np.ma.masked_equal(mask_raw, 0))
        self.front, self.mask_front, self.add_front, mask_raw = self.add_front_image(mask_raw)
        meta = scio.loadmat(self.paths['meta'][index])
        depth = depth_read(self.paths['depth'][index], meta['factor_depth'][0][0]) 

        obj = meta['cls_indexes'].flatten().astype(np.int32)
        mask, mask_label, idx = self.preprocess_mask(mask_raw, depth, obj)

        dataloader_id = self._to_dataloader_id(obj[idx])

        bbox = get_bbox_ycb(mask_label, self.img_height, self.img_width)
        pose = np.array(meta['poses'][:, :, idx])
        
        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            self.matrix_K = self.matrix_K2
            self.cam_fx = self.cam_fx_2
            self.cam_fy = self.cam_fy_2
            self.cam_cx = self.cam_cx_2
            self.cam_cy = self.cam_cy_2
        else:
           pass
 
        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = back_read(os.path.join(self.root_dir, seed+'-color.png'))
            rgb += back*mask_back[:, :, None]

        candidates = {
           'rgb': rgb,
           'depth': depth,
           'mask': mask,
           'bbox': bbox,
           'pose': pose,
           'model': self.models[dataloader_id], # check if the index is correct
        }
        return candidates 

    def preprocess(self, index, raws):
        '''extra processing after calling __getraw__
        '''
        ymin, ymax, xmin, xmax = raws['bbox']
        bbox_index = np.s_[ymin:ymax, xmin:xmax]

        rgb_masked = raws['rgb'][bbox_index]
        # depth_masked = raws['depth'][bbox_index]
        mask_in_bbox = self.get_mask_inside_bbox(raws['mask'], bbox_index)

        pts_sensor_cam = self.get_pts_sensor_cam(raws['depth'], bbox_index, 
            mask_in_bbox)
        model_points = raws['model'].get_model_points()
        raw_model_points = raws['model'].get_raw_model_points()
        model_index = raws['model'].get_index()
        pts_model_cam = self.model2cam(model_points, raws['pose'])

        pts_farthest = raws['model'].get_farthest_points()
        pts_farthest_cam = self.model2cam(pts_farthest, raws['pose'])
        pts_farthest_img = self.model2img(pts_farthest, raws['pose'])

        gt_dir_vecs_c = self.get_gt_vector_field(pts_farthest_cam, pts_sensor_cam)
        rgb_selected = rgb_masked.flatten()[mask_in_bbox][:, np.newaxis].astype(np.float32)
        gt_dir_vecs_i = self.get_gt_vector_field(pts_farthest_img, rgb_selected)


        
        if self.add_noise and self.add_front:
            print('1')
            rgb_masked = rgb_masked * self.mask_front[ymin:ymax, xmin:xmax, None] \
                + self.front[ymin:ymax, xmin:xmax, :] * ~(self.mask_front[ymin:ymax, xmin:xmax, None])

        if self.list[index][:8] == 'data_syn':
            rgb_masked = rgb_masked + np.random.normal(loc=0.0, scale=7.0, size=rgb_masked.shape)
        
        # self.draw_vector_field_3D(pts_sensor_cam, gt_dir_vecs_c[:, 0, :])
        items = {
            'rgb': raws['rgb'],
            'depth': raws['depth'], # depth of whole image
            'rgb_masked': rgb_masked,
            # 'depth_masked': depth_masked,
            'pts_sensor_cam': pts_sensor_cam, # observed point cloud of object in camera frame
            'pose': raws['pose'],
            'mask_in_bbox': mask_in_bbox,
            'model_points': model_points,   # 3d points of object model
            'raw_model_points': raw_model_points,
            'pts_model_cam': pts_model_cam, # 3d points of object model in camera frame
            'model_index': model_index,
            'bbox': np.array([ymin, ymax, xmin, xmax]),
            'gt_3d_vector_field': gt_dir_vecs_c,
            'gt_2d_vector_field': gt_dir_vecs_i,
            'pts_farthest_model': pts_farthest, 
            'pts_farthest_cam': pts_farthest_cam
        }
        return items

    
