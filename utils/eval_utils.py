import numpy as np
import torch

from models.knn.__init__ import KNearestNeighbor
from utils.tf_numpy import euler_matrix, \
    quaternion_matrix, quaternion_from_matrix, \
    quaternion_distance, quaternion_angle

knn = KNearestNeighbor(1)

def compute_adds_metric(best_rot, trans_pred, rot_gt, trans_gt, 
      model_points, is_symmetric):
    '''
    compute the mean of closest point distance
    '''
    rot_gt = rot_gt
    trans_gt = trans_gt
    assert best_rot.shape == (3,3)
    assert trans_pred.shape == (3,1)
    assert rot_gt.shape == (3,3)
    assert trans_gt.shape == (3,1)

    model_points = model_points
    model_pred = np.dot(model_points, best_rot.T) + trans_pred.T
    model_gt = np.dot(model_points, rot_gt.T) + trans_gt.T
    if is_symmetric:
        model_pred = torch.from_numpy(
            model_pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        model_gt = torch.from_numpy(
            model_gt.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn.apply(1, model_gt.unsqueeze(0), model_pred.unsqueeze(0))
        model_gt = torch.index_select(model_gt, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((
            model_pred.transpose(1, 0) 
            - model_gt.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

    return dis

def is_correct_pred(dis_pred, model_diameter, threshold=0.1):
    '''
    The 6D pose is considered to be correct if the average distance 
        is smaller than a predefined threshold (default is 10%)
    '''
    return dis_pred < model_diameter*threshold

def compute_error(best_quat, best_t, gt_quat, gt_t):
    assert best_t.shape == (3, 1)
    assert gt_t.shape == (3, 1)
    x_offset, y_offset, z_offset = abs(best_t - gt_t)
    q_distance = quaternion_distance(best_quat, gt_quat)
    q_angle = quaternion_angle(best_quat, gt_quat)
    return q_distance, q_angle, x_offset[0], y_offset[0], z_offset[0]

class IoU:
    r"""Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

        # res: stored results, col 0: sum; col 1: # of cases; col 2: average
        self.res = np.zeros((num_classes, 3))

    def add(self, pred_label, gt_label, class_id):
        r"""Add one test case
        args:
            pred_label: (H, W)
            gt_label: (H, W)
        """
        iou = np.count_nonzero(pred_label+gt_label>1) \
            / (np.count_nonzero(pred_label) \
            + np.count_nonzero(gt_label))

        self.res[class_id, 0] += iou
        self.res[class_id, 1] += 1
        self.res[class_id, 2] = self.res[class_id, 0] / self.res[class_id, 1]
