import os.path
import numpy as np
import random
import open3d as o3d

from dataloader.data_utils import *


class ObjectModel:
    def __init__(self, root_dir, idx, index, name, num_points):
        self.root_dir = root_dir
        self.id = idx # 
        self.index = index # index is continous
        self.name = name
        self.num_points = num_points # downsample ply file point

        self.diameter = None
        self.model_points = None 
        self.model_path = None # path to the model

        self.corners = None # corner points of model points
        self.center = None
        self.farthest_points = None

    def get_id(self):
        return self.id

    def get_index(self):
        return self.index

    def get_model_points(self):
        return self.model_points

    def get_raw_model_points(self):
        return self.raw_model_points

    def get_model_diameter(self):
        return self.diameter

    def is_symmetric(self):
        return self.is_sym
        
    def _downsample_points(self, num_points):
        '''downsample 3D points
        args: 
            num_points: number of points after sampling
        '''
        # np.random.seed(0)
        dellist = [j for j in range(0, len(self.raw_model_points))]
        dellist = random.sample(dellist, len(self.raw_model_points) - num_points)
        # dellist = dellist[:len(self.raw_model_points)-num_points] # remove randomness, use for test
        downsampled_model = np.delete(self.raw_model_points, dellist, axis=0)
        return downsampled_model

    def load_model(self, root_dir, unit_scale=1000.0):
        self.raw_model_points = self._read_model_pts()
        self.raw_model_points /= unit_scale # convert to meter; for linemod, it's 1000; for ycb, its 1
        # print(self.num_points)
        self.model_points = self._downsample_points(self.num_points)

    def compute_farthest_surface_point(self, num_pt, write_to_file=True):
        from dataloader.extend_utils.extend_utils import farthest_point_sampling

        if self.model_points is None:
            print('model points are not loaded')
            return
        farthest_pts = farthest_point_sampling(self.model_points, num_pt, True)
        if write_to_file:
            filename = 'obj_%02d_farthest%02d.txt' % (self.id, num_pt)
            path = os.path.join(self.model_path, 'farthest_surface_point')
            if not os.path.exists(path):
                os.makedirs(path)
            write_points(os.path.join(path, filename), farthest_pts)

    def get_corner_points(self):
        if self.corners is not None:
            return self.corners

        if self.model_points is None:
            raise Exception('Object model is not initialized!')

        x = self.model_points[:, 0]
        min_x, max_x = np.min(x), np.max(x)
        y = self.model_points[:, 1]
        min_y, max_y = np.min(y), np.max(y)
        z = self.model_points[:, 2]
        min_z, max_z = np.min(z), np.max(z)
        self.corners = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return self.corners

    def get_center(self):
        '''return the center point of the model
        '''
        if self.center is not None:
            return self.center

        corners = self.get_corner_points()
        self.center=(np.max(corners,0)+np.min(corners,0))/2
        return self.center

    def get_farthest_points(self, num_pt=8):
        '''get #num_pt farthest points on the model surface
        '''
        if self.farthest_points is not None \
            and self.farthest_points.shape[0] == num_pt:
            return self.farthest_points

        path = os.path.join(self.model_path, 'farthest_surface_point')
        farthest_point_file = 'obj_%02d_farthest%02d.txt' % (self.id, num_pt)
        self.farthest_points = np.loadtxt(os.path.join(path, farthest_point_file))
        
        return self.farthest_points

    def show_model(self, num_pt):
        '''visualize point cloud 
        args:
            num_pt: number of farthest points
        '''
        if self.model_points is None:
            return

        path = os.path.join(self.model_path, 'farthest_surface_point')
        farthest_point_file = 'obj_%02d_farthest%02d.txt' % (self.id, num_pt)
        ply_file = 'obj_%02d.ply' % (self.id)

        farthest_points = np.loadtxt(os.path.join(path, farthest_point_file))
        farthest_colors = np.tile(np.array([255,0,0]), (num_pt,1))

        np_colors = np.zeros_like(self.model_points) #+ np.array([255, 255, 255])
        # vis_points = np.vstack( (self.model_points, farthest_points) )
        # vis_colors = np.vstack( (np_colors, farthest_colors) )
        vis_points = farthest_points
        vis_colors = farthest_colors

        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(vis_points)
        pcd.colors = o3d.Vector3dVector(vis_colors)
        o3d.draw_geometries([pcd]) # Visualize the point cloud 