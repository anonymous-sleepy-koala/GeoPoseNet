import os
import numpy as np
import open3d as o3d
import random
import copy
from scipy.linalg import expm, norm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------
# Data IO
# ---------------------------

def rgb_read(filename):
    # rgb_image = np.array(Image.open(filename))[:, :, :3] #/ 255.0
    rgb_image = plt.imread(filename) #/ 255.0
    return rgb_image

def depth_read(filename, depth_factor=1000.0):
    depth_image = np.array(Image.open(filename)) / depth_factor # convert to meter
    return depth_image

def mask_read(filename):
    mask_image = np.array(Image.open(filename))
    return mask_image

def seg_mask_read(mask_path):
    '''
    return:
        seg_mask: (H, W)
    '''
    seg_mask = np.array(Image.open(mask_path)).astype(np.int32)
    if len(seg_mask.shape) == 3:
        seg_mask=np.sum(seg_mask, 2)>0
    elif len(seg_mask.shape) == 2:
        seg_mask=seg_mask>0
    else:
        raise NotImplementedError('unrecognized data shape')
    seg_mask=np.asarray(seg_mask,np.int32)
    return seg_mask

def back_read(filename):
    return np.array(Image.open(filename).convert("RGB")) / 255.0

# ---------------------------
# End of Data IO
# ---------------------------


# ---------------------------
# Point Cloud Sampling
# ---------------------------

def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0)) # move the mean to the origin
  T[:3, 3] = 3*(np.random.rand(3) - 0.5)
  return T


def sample_3d_vector():
    theta = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1, 1)
    return np.array([np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z])

def sample_knn(pcd, K, ref_pt='center'):
    r"""Randomly sample K nearest neighbors to a point, from a given point cloud pcd
    args:
        ref_pt: the reference point. If ref_pt='center',
            we choose the center of the point cloud as the reference point.
            otherwise, we take the center point as starting point, randomly sample a
            direction vector in 3D, extend r=5m along the direction vector to obtain
            the reference point.
    """
    if ref_pt == 'center':
        ref = np.mean(pcd, axis=0)
    else:
        direction = sample_3d_vector()
        r = 10
        ref = np.mean(pcd, axis=0) + r * direction
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(pcd)
    _, ind = neigh.kneighbors(np.array([ref]))
    return pcd[ind][0]
    
def sample_partial_pcd(pcd):
    r"""Randomly generate a plane which passes the center of a point cloud,
    return points that are on one side of the plane
    
    args:
        pcd: (num_ponts, 3), input point cloud
    """
    center = np.mean(pcd, axis=0)
    normal = sample_3d_vector()
    
    sign = np.dot(pcd - center, normal)    
    return pcd[sign > 0]

def uniform_sample_pcd(pcd, num_sample):
    '''downsample 3D points
    args: 
        num_points: number of points after sampling
    '''
    # np.random.seed(0)
    num_pts = pcd.shape[0]
    print(num_sample)
    dellist = [j for j in range(0, num_pts)]
    dellist = random.sample(dellist, num_pts - num_sample)
    # dellist = dellist[:len(self.raw_model_points)-num_points] # remove randomness, use for test
    downsampled_pcd = np.delete(pcd, dellist, axis=0)
    return downsampled_pcd

# ---------------------------
# End of Point Cloud Sampling
# ---------------------------

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def make_open3d_point_cloud(xyz, color=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  if color is not None:
    pcd.colors = o3d.utility.Vector3dVector(color)
  return pcd


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def get_matching_indices_no_trans(source, target, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    # source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def compute_overlap_ratio(pcd0, pcd1, voxel_size, match_thresh_ratio=1):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    # print(np.asarray(pcd0_down.points).shape, np.asarray(pcd0.points).shape)
    # print('1',np.asarray(pcd1_down.points).shape, np.asarray(pcd1.points).shape)
    matching01 = get_matching_indices_no_trans(pcd0_down, pcd1_down, 
        match_thresh_ratio*voxel_size, 1)
    matching10 = get_matching_indices_no_trans(pcd1_down, pcd0_down,
        match_thresh_ratio*voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    print(overlap0, overlap1)
    return max(overlap0, overlap1)

def write_points(filename, pts, colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(
                        pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if colors.shape[0]==pts.shape[0]:
                    f.write('{} {} {} {} {} {}\n'.format(
                        pt[0],pt[1],pt[2],
                        int(colors[i,0]),
                        int(colors[i,1]),
                        int(colors[i,2])))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(
                        pt[0],pt[1],pt[2],
                        int(colors[0]),
                        int(colors[1]),
                        int(colors[2])))

# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
# def get_bbox_linemod(bbox):
#     bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
#     if bbx[0] < 0:
#         bbx[0] = 0
#     if bbx[1] >= 480:
#         bbx[1] = 479
#     if bbx[2] < 0:
#         bbx[2] = 0
#     if bbx[3] >= 640:
#         bbx[3] = 639                
#     rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
#     r_b = rmax - rmin
#     for tt in range(len(border_list)):
#         if r_b > border_list[tt] and r_b < border_list[tt + 1]:
#             r_b = border_list[tt + 1]
#             break
#     c_b = cmax - cmin
#     for tt in range(len(border_list)):
#         if c_b > border_list[tt] and c_b < border_list[tt + 1]:
#             c_b = border_list[tt + 1]
#             break
#     center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
#     rmin = center[0] - int(r_b / 2)
#     rmax = center[0] + int(r_b / 2)
#     cmin = center[1] - int(c_b / 2)
#     cmax = center[1] + int(c_b / 2)
#     if rmin < 0:
#         delt = -rmin
#         rmin = 0
#         rmax += delt
#     if cmin < 0:
#         delt = -cmin
#         cmin = 0
#         cmax += delt
#     if rmax > 480:
#         delt = rmax - 480
#         rmax = 480
#         rmin -= delt
#     if cmax > 640:
#         delt = cmax - 640
#         cmax = 640
#         cmin -= delt
#     return rmin, rmax, cmin, cmax

# def get_bbox_ycb(label, img_height, img_width):
#     rows = np.any(label, axis=1)
#     cols = np.any(label, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     rmax += 1
#     cmax += 1
#     r_b = rmax - rmin
#     for tt in range(len(border_list)):
#         if r_b > border_list[tt] and r_b < border_list[tt + 1]:
#             r_b = border_list[tt + 1]
#             break
#     c_b = cmax - cmin
#     for tt in range(len(border_list)):
#         if c_b > border_list[tt] and c_b < border_list[tt + 1]:
#             c_b = border_list[tt + 1]
#             break
#     center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
#     rmin = center[0] - int(r_b / 2)
#     rmax = center[0] + int(r_b / 2)
#     cmin = center[1] - int(c_b / 2)
#     cmax = center[1] + int(c_b / 2)
#     if rmin < 0:
#         delt = -rmin
#         rmin = 0
#         rmax += delt
#     if cmin < 0:
#         delt = -cmin
#         cmin = 0
#         cmax += delt
#     if rmax > img_height:
#         delt = rmax - img_height
#         rmax = img_height
#         rmin -= delt
#     if cmax > img_width:
#         delt = cmax - img_width
#         cmax = img_width
#         cmin -= delt
#     return rmin, rmax, cmin, cmax

# def get_bbox_posecnn(posecnn_rois):
#     rmin = int(posecnn_rois[idx][3]) + 1
#     rmax = int(posecnn_rois[idx][5]) - 1
#     cmin = int(posecnn_rois[idx][2]) + 1
#     cmax = int(posecnn_rois[idx][4]) - 1
#     r_b = rmax - rmin
#     for tt in range(len(border_list)):
#         if r_b > border_list[tt] and r_b < border_list[tt + 1]:
#             r_b = border_list[tt + 1]
#             break
#     c_b = cmax - cmin
#     for tt in range(len(border_list)):
#         if c_b > border_list[tt] and c_b < border_list[tt + 1]:
#             c_b = border_list[tt + 1]
#             break
#     center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
#     rmin = center[0] - int(r_b / 2)
#     rmax = center[0] + int(r_b / 2)
#     cmin = center[1] - int(c_b / 2)
#     cmax = center[1] + int(c_b / 2)
#     if rmin < 0:
#         delt = -rmin
#         rmin = 0
#         rmax += delt
#     if cmin < 0:
#         delt = -cmin
#         cmin = 0
#         cmax += delt
#     if rmax > img_width:
#         delt = rmax - img_width
#         rmax = img_width
#         rmin -= delt
#     if cmax > img_length:
#         delt = cmax - img_length
#         cmax = img_length
#         cmin -= delt
#     return rmin, rmax, cmin, cmax

# def quantize_value(x):
#     x = int( int(x / 40.0+1) * 40 )
#     return x

# def clip_bbox(xc, yc, width, height, img_width, img_height):
#     '''
#     args:
#         xc, yc: center of bbox (type: int)
#         width: image width (type: even int)
#         height: image height (type: even int)
#     '''
#     ymin = yc - height // 2
#     ymax = yc + height // 2 - 1
#     xmin = xc - width // 2
#     xmax = xc + width // 2 - 1
#     if ymin < 0:
#         delt = -ymin
#         ymin = 0
#         ymax = ymax + delt
#     if ymax >= img_height:
#         delt = ymax - img_height + 1
#         ymax = img_height - 1
#         ymin -= delt
#     if xmin < 0:
#         delt = -xmin
#         xmin = 0
#         xmax += delt
#     if xmax >= img_width:
#         delt = xmax - img_width + 1
#         xmax = img_width - 1
#         xmin -= delt
#     return ymin, ymax, xmin, xmax

# def get_bbox(input_bbox, img_width, img_height):
#     '''
#     args:
#         input_bbox: [xmin, ymin, width, height]
#     '''
#     ymin = input_bbox[1]
#     ymax = input_bbox[1] + input_bbox[3] - 1
#     xmin = input_bbox[0]
#     xmax = input_bbox[0] + input_bbox[2] - 1
#     ymin, ymax, xmin, xmax = np.clip([ymin, ymax, xmin, xmax], 
#         0, [img_height-1, img_height-1, img_width-1, img_width-1])
    
#     yc = int((ymin + ymax) / 2)
#     xc = int((xmin + xmax) / 2)
#     width_bbox = quantize_value(xmax-xmin+1)
#     height_bbox = quantize_value(ymax-ymin+1)

#     return clip_bbox(xc, yc, width_bbox, height_bbox, img_width, img_height)


