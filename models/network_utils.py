import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.tf_numpy as tf_numpy
import utils.tf_torch as tf

import time
# ----------------------------------------------
def least_square_intersection(pts_src, dir_vecs):
    norm = torch.norm(dir_vecs, p=2, dim=1, keepdim=True).detach()
    dn = dir_vecs.div(norm.expand_as(dir_vecs)) # normalized direction vector

    projs = torch.eye(pts_src.shape[1]) \
        - torch.mul(dir_vecs[:, :, None], dir_vecs[:, None, :])

    R = torch.sum(projs, dim=0)
    a = torch.matmul(projs, pts_src[:, :, None])
    q = torch.sum(torch.matmul(projs, pts_src[:, :, None]), dim=0)

    betas_qr,_ = torch.gels(q,R)
    return betas_qr

class SVDTransform(nn.Module):
    '''Adapt from https://github.com/WangYueFt/dcp/blob/master/model.py
    '''
    def __init__(self, device):
        super(SVDTransform, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3, device=device), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, tgt):
        bs, _, _ = src.shape
        src = src.permute(0, 2, 1)
        tgt = tgt.permute(0, 2, 1)

        src_centered = src - src.mean(dim=2, keepdim=True)
        tgt_centered = tgt - tgt.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, tgt_centered.transpose(2, 1).contiguous())

        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:                
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + tgt.mean(dim=2, keepdim=True)
        return R, t.view(bs, 3)
# ----------------------------------------------

def distance_from_point2vec(point, pt_src, dir_vec):
    norm_ = np.linalg.norm(np.cross(dir_vec, pt_src-point), axis=1)
    return np.linalg.norm(np.cross(dir_vec, pt_src-point), axis=1)/np.linalg.norm(dir_vec)

def get_num_inliers(pt_intersect, pts_src, pts_dir, epsilon=0.0005):
    r"""
    args:
        pt_intersect: (1, 3)
        pts_src: (num_pts, 3)
        pts_dir: (num_pts, 3)
    """
    num_inliers = 0
    distance = distance_from_point2vec(pt_intersect, pts_src, pts_dir)

    return np.count_nonzero(distance < epsilon), distance<epsilon

def get_angle_offset(pts_src, pred_dirs, gt):
    r"""
    args:
        pts_src: (num_pts, 3)
        pred_dirs: (num_pts, 3)
        gt: (1, 3)
    """
    temp = np.einsum('ij,ij->i',gt, pred_dirs) / np.linalg.norm(pred_dirs, axis=1) #np.tensordot(gt, pred_dirs, axes=1)
    # print(gt.shape, temp.shape)
    return temp

def get_dir_pred_distribution(pts_sensor_cam, pred_dirs, pts_gt_farthest):
    """
    args:
        pts_sensor_cam: (bs, num_pts, 3)
        pred_dirs: (bs, num_pts, num_farthest_pts, 3)
        pts_gt_farthest: (bs, num_pts, num_farthest_pts, 3)
    """
    bs, num_pts, num_farthest_pts, _ = pred_dirs.shape
    slots = {0.99:0, 0.95:0, 0.9:0, 0.8:0, 0.5:0}
    for i in range(bs):
        for j in range(num_farthest_pts):
            cos_theta = get_angle_offset(pts_sensor_cam[i, :, :], 
                pred_dirs[i, :, j, :], pts_gt_farthest[i, :, j, :])
            slots[0.99] += np.count_nonzero(cos_theta > 0.99)
            slots[0.95] += np.count_nonzero(cos_theta > 0.95)
            slots[0.9] += np.count_nonzero(cos_theta > 0.9)
            slots[0.8] += np.count_nonzero(cos_theta > 0.8)
            slots[0.5] += np.count_nonzero(cos_theta > 0.5)
    return slots

def get_farthestpts_pred_distribution(pred_farthest_pts, gt_farthest_pts):
    """
    args:
        pred_farthest_pts: (bs, num_farthest_pts, 3)
        gt_farthest_pts: (bs, num_farthest_pts, 3)
    """
    bs, num_farthest_pts, _ = pred_farthest_pts.shape
    slots = {0.007:0, 0.002:0, 0.005:0, 0.01:0, 0.02:0}
    for i in range(bs):
        for j in range(num_farthest_pts):
            cos_theta = np.linalg.norm( pred_farthest_pts[i, j, :]- gt_farthest_pts[i, j, :] ) 
            
            slots[0.007] += 1 if cos_theta < 0.007 else 0
            slots[0.002] += 1 if cos_theta < 0.002 else 0
            slots[0.005] += 1 if cos_theta < 0.005 else 0
            slots[0.01] += 1 if cos_theta < 0.01 else 0
            slots[0.02] += 1 if cos_theta < 0.02 else 0
    return slots


def get_closest_point_stochastic(pts_sensor_cam, pred_dirs, pred_conf):
    bs, num_pts, num_farthest_pts, _ = pred_dirs.shape
    pred_farthest_pts = np.zeros((bs, num_farthest_pts, 3))
    K = -100
    for i in range(bs):
        for j in range(num_farthest_pts):
            topN_index = np.argpartition(pred_conf[i, :, j, :], -K, axis=0)[-K:][:, 0]
            # topN_index = np.random.choice(num_pts, 100)
            # print(topN_index)
            pred_farthest_pts[i, j, :] = tf.line_intersection(
                pts_sensor_cam[i, :][topN_index, :],
                pred_dirs[i, :, j, :][topN_index, :])

    return pred_farthest_pts

def get_closest_point_ls(pts_sensor_cam, pred_dirs):
    bs, num_pts, num_farthest_pts, _ = pred_dirs.shape
    pred_farthest_pts = np.zeros((bs, num_farthest_pts, 3))
    for i in range(bs):
        for j in range(num_farthest_pts):
            #print(pred_dirs.detach()[i, :, j, :])
            pred_farthest_pts[i, j, :] = tf.line_intersection(
                pts_sensor_cam[i, :].cpu().numpy(), 
                pred_dirs.detach()[i, :, j, :].cpu().numpy() )
    return pred_farthest_pts

def get_closest_point_ransac(pts_src, dir_vecs):
    '''
    args:
        pts_src: (bs, num_pts, 3), starting points of the vector field
        dir_vecs: (bs, num_pts, num_farthest_pts, 3), direction vector of each starting point
    '''
    bs, num_pts, num_farthest_pts, _ = dir_vecs.shape
    pred_farthest_pts = np.zeros((bs, num_farthest_pts, 3))
    for i in range(bs):
        for j in range(num_farthest_pts):
            pts_cur_src = pts_src[i, :, :]
            pts_cur_dir = dir_vecs[i, :, j, :]
            max_num_inlier = 0
            num_iter = 0
            best_pt = None
            best_inlier_index = None
            while num_iter < 50:
                num_iter += 1
                index = np.random.choice(num_pts, 2)
                if index[0] == index[1]:
                    continue
                pt_intersect = tf.line_intersection(
                    pts_cur_src[index, :], pts_cur_dir[index, :])
                num_inliers, inlier_index = get_num_inliers(pt_intersect, pts_cur_src, pts_cur_dir)
                if num_inliers > max_num_inlier:
                    max_num_inlier = num_inliers
                    best_pt = pt_intersect
                    best_inlier_index = inlier_index
            pt_intersect = tf.line_intersection(
                    pts_cur_src[best_inlier_index, :], pts_cur_dir[best_inlier_index, :])
            pred_farthest_pts[i, j, :] = pt_intersect 
    return pred_farthest_pts

def get_pose_pred(pts_sensor_cam, pred_dirs, pts_farthest_model, device, pred_conf=None):
    '''
    args:
        pts_sensor_cam: (bs, num_pts, 3)
        pred_dirs: (bs, num_pts, num_keypoints, 3)
        pred_conf: (bs, num_pts, num_keypoints, 3), uncerntainty of prediction
        pts_farthest_model: (bs, num_keypoints, 3)
    '''
    bs, num_pts, _ = pts_sensor_cam.shape
    alg = 'least_square'
    # alg = 'ransac'
    # alg = 'stochastic'
    if alg == 'least_square':
        pred_farthest_pts = get_closest_point_ls(pts_sensor_cam, 
            pred_dirs)
    elif alg == 'ransac':
        pred_farthest_pts = get_closest_point_ransac(pts_sensor_cam.cpu().numpy(), 
            pred_dirs.cpu().numpy())
    elif alg == 'stochastic':
        pred_farthest_pts = get_closest_point_stochastic(pts_sensor_cam.cpu().numpy(), 
            pred_dirs.detach().cpu().numpy(), pred_conf.detach().cpu().numpy())
    else:
        raise NotImplementedError('[%s] is not implemented', alg)
    # svd
    # rot_mat, t_vec = best_fit_transform(pts_farthest_model,
    #     torch.from_numpy(pred_farthest_pts[None, ].copy()).float().to(device))
    # quat = quaternion_from_matrix(rot_mat[0].cpu().numpy())
    # quat = torch.from_numpy(quat.copy()).float().to(device)
    # return rot_mat.detach(), t_vec.detach(), quat
    
    # numpy version
    rot_mat, t_vec = tf.rigid_transform_3D(pts_farthest_model[0, :, :].cpu().numpy(),
        pred_farthest_pts[0, :, :])
    quat = tf_numpy.quaternion_from_matrix(rot_mat)
    
    rot_mat = torch.from_numpy(rot_mat.copy()).float().to(device)
    t_vec = torch.from_numpy(t_vec.copy()).float().to(device)
    quat = torch.from_numpy(quat.copy()).float().to(device)

    return rot_mat.detach(), t_vec.view(1, 3).detach(), quat.view(1, 4).detach(), pred_farthest_pts # 3x3, 1x3, 1x4

def get_pose_pred_batch(pts_sensor_cam, pred_dirs, pts_farthest_m, len_batch, device=None):
    '''
    args:
        pts_sensor_cam: (num_pts, 4)
        pred_dirs: (num_pts, num_keypoints, 3)
        pts_farthest_model: (bs, num_keypoints, 3)
    '''
    DEBUG = True
    if DEBUG:
        num_pts = pts_sensor_cam.shape[0]
        assert pts_sensor_cam.shape == torch.Size([num_pts, 3]) 
        assert pred_dirs.shape == torch.Size([num_pts, 8, 3])
        assert pts_farthest_m.shape == torch.Size([8, 3])
      
    st = 0
    pred_rot = []
    pred_t = []
    for i, num in enumerate(len_batch):
        num = num[0]
        pts_sensor = pts_sensor_cam[st:st+num, :3]
        dirs = pred_dirs[st:st+num, :, :]
        pts_farthest = pts_farthest_m[i*8:i*8+8, :]
       
        start = time.time() 
        pred_farthest_pts = get_closest_point_ls(
            pts_sensor[None, :], 
            dirs[None, :])
        end = time.time()
        t1 = end - start
        start = end

        st += num
    
        # numpy version
        rot_mat, t_vec = tf.rigid_transform_3D(pts_farthest.cpu().numpy(),
            pred_farthest_pts[0, :, :])
        t2 = time.time() - start
        pred_rot.append(rot_mat)
        pred_t.append(t_vec)
    return pred_rot, pred_t, t1, t2 # 3x3, 3x1


#def get_pose_pred_batch(pts_sensor_cam, pred_kpts, pts_farthest_m, len_batch, device=None):
#    '''
#    args:
#        pts_sensor_cam: (num_pts, 4)
#        pred_kpts: (bs, num_keypoints, 3)
#        pts_farthest_m: (bs, num_keypoints, 3)
#    '''
#    st = 0
#    pred_rot = []
#    pred_t = []
#    for i, num in enumerate(len_batch):
#        num = num[0]
#        #pts_sensor = pts_sensor_cam[st:st+num, :3]
#        #dirs = pred_dirs[st:st+num, :, :]
#        pts_farthest = pts_farthest_m[i*8:i*8+8, :]
#        pred_farthest_pts = pred_kpts[i*8:i*8+8, :]
#        #print('debug 3, pred_farthest_pts', pred_farthest_pts.shape, pts_farthest.shape)
#        
#        #pred_farthest_pts = get_closest_point_ls(
#        #    pts_sensor[None, :], 
#        #    dirs[None, :])
#
#        st += num
#    
#        # numpy version
#        rot_mat, t_vec = tf.rigid_transform_3D(pts_farthest.cpu().numpy(),
#            pred_farthest_pts[:, :].cpu().numpy())
#        pred_rot.append(rot_mat)
#        pred_t.append(t_vec)
#    return pred_rot, pred_t # 3x3, 3x1
