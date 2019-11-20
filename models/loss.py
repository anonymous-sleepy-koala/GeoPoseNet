import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

from models.knn.__init__ import KNearestNeighbor
import utils.tf_numpy as tf_numpy 
import utils.tf_torch as tf

class StoLoss(nn.Module):
    '''Define L1 loss of predicted stochastic direction filed 

    '''
    def __init__(self):
        super(StoLoss, self).__init__()
        self.l1 = nn.L1Loss()
    
    def __call__(self, pred_vector_field, pred_conf, gt_vector_field):
        r"""
        args:
            pred_vector_filed: (bs, num_pts, num_keypoints, 3)
            pred_conf: (bs, num_pts, num_keypoints, 1)
            gt_vector_field: (bs, num_pts, num_keypoints, 3)
        """
        w = 0.015
        dis = torch.mean(torch.norm(pred_vector_field - gt_vector_field, dim=3))
        pred_conf = torch.flatten(pred_conf)
        return torch.mean((dis * pred_conf - w * torch.log(pred_conf)))

class DistanceLoss(nn.Module):
    '''Define loss based on average distance between two point clouds

    '''
    def __init__(self, sym_list):
        super(DistanceLoss, self).__init__()
        self.sym_list = sym_list

    def __call__(self, pred_R, pred_t, pts_model, pts_gt, model_index, device):
        r"""Calculate loss given predicted pose [pred_quat, pred_t]

        args:
            pred_R: (bs, 3, 3) predicted rotation
            pred_t: (bs, 3) predicted translation
            pts_model: (bs,num_pts,3), 3D points in model frame
            pts_gt: (bs,num_pts,3),3D points in cam frame (ground truth, 
              obtained by applying ground truth pose to pts_model)
            model_index: index of the object model
            device: gpu id
        """
        bs, num_pts, _ = pts_model.shape     
        assert pred_R.shape == torch.Size([bs, 3, 3])
        assert pred_t.shape == torch.Size([bs, 3])
        
        pose_pred = tf.assemble_pose(pred_R, pred_t, device)
        pts_pred = tf.transform_points(pose_pred, pts_model) #(bs, num_pts, 3)

        if model_index.item() in self.sym_list:
            knn = KNearestNeighbor(1)
            pts_gt = pts_gt[0].transpose(1, 0).contiguous().view(3, -1)
            pts_pred = pts_pred.permute(2, 0, 1).contiguous().view(3, -1)
            inds = knn(pts_gt.unsqueeze(0), pts_pred.unsqueeze(0))
            pts_gt = torch.index_select(pts_gt, 1, inds.view(-1) - 1)

            pts_gt = pts_gt[None, :, :].permute(0, 2, 1)
            pts_pred = pts_pred[None, :, :].permute(0, 2, 1)

            del knn
        
        dis_loss = torch.mean(torch.norm((pts_pred - pts_gt), dim=2), dim=1)
        return dis_loss

class ADDSLoss(nn.Module):
    '''Define loss based on average distance between two point clouds

    '''
    def __init__(self, sym_list):
        super(ADDSLoss, self).__init__()
        self.sym_list = sym_list

    def __call__(self, pred_R, pred_t, pts_model, pose_gt, model_index, device):
        r"""Calculate loss given predicted pose [pred_quat, pred_t]

        args:
            pred_R: (bs, 3, 3) predicted rotation
            pred_t: (bs, 3) predicted translation
            pts_model: (bs,num_pts,3), 3D points in model frame
            pts_gt: (bs,num_pts,3),3D points in cam frame (ground truth, 
              obtained by applying ground truth pose to pts_model)
            model_index: index of the object model
            device: gpu id
        """
        # pts_model = torch.from_numpy(pts_model[None, :])
        bs, num_pts, _ = pts_model.shape  
        pose_gt = pose_gt[None, :]   
        # print('pts model', pts_model.shape)
        # print('R', pred_R.shape)
        # print('t', pred_t.shape)
        # print('pose gt', pose_gt.shape)
        #assert pred_R.shape == torch.Size([bs, 3, 3])
        assert pred_t.shape == torch.Size([bs, 3])
        
        # pose_pred = tf.assemble_pose(pred_R, pred_t, device)
        # print('pose_pred', pose_pred.shape)
        pts_pred = tf.transform_pts(pred_R, pred_t, pts_model) #(bs, num_pts, 3)
        # pts_pred = tf.transform_points(pose_pred, pts_model) #(bs, num_pts, 3)


        pts_gt = tf.transform_points(pose_gt, pts_model)
        # print('ahh',pts_gt.shape)

        # if model_index in self.sym_list:
        #     knn = KNearestNeighbor(1)
        #     pts_gt = pts_gt[0].transpose(1, 0).contiguous().view(3, -1)
        #     pts_pred = pts_pred.permute(2, 0, 1).contiguous().view(3, -1)
        #     inds = knn(pts_gt.unsqueeze(0), pts_pred.unsqueeze(0))
        #     pts_gt = torch.index_select(pts_gt, 1, inds.view(-1) - 1)

        #     pts_gt = pts_gt[None, :, :].permute(0, 2, 1)
        #     pts_pred = pts_pred[None, :, :].permute(0, 2, 1)

        #     del knn
        
        dis_loss = torch.mean(torch.norm((pts_pred - pts_gt), dim=2), dim=1)
        return dis_loss
