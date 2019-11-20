import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from utils.transformations import quaternion_matrix, quaternion_from_matrix

from options.test_options import TestOptions
from models import create_model



args = TestOptions().parse()
# visualizer = Visualizer(args)
model = create_model(args) 
model.setup(args) 


from dataloader.ycb_dataloader import YCBVideo
test_set = YCBVideo(args.data_path, 'test', False, args.num_points)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, 
    shuffle=False, num_workers=args.workers, sampler=None)


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'dataloader/config/ycb'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_refine_dir = os.path.join(args.checkpoints_dir, args.name, 'refine_res')
result_wo_refine_dir = os.path.join(args.checkpoints_dir, args.name, 'wo_refine_res')
def icp_refine(R, t, model_points, pts_sensor_cam):
    source = o3d.PointCloud()
    source.points = o3d.Vector3dVector(model_points[0, :].cpu().numpy())
    target = o3d.PointCloud()
    target.points = o3d.Vector3dVector(pts_sensor_cam[0,:].cpu().numpy())

    threshold = 0.01
    temp = np.concatenate((R, t), axis=1)
   
    trans_init = np.concatenate((temp, np.array([[0.0, 0.0, 0.0, 1.0]])))

    # draw_registration_result(source, target, trans_init)
    evaluation = o3d.registration.evaluate_registration(source, target,
                                            threshold, trans_init)
    # print("Initial alignment")
    # print(evaluation)

    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    
    
    R = reg_p2p.transformation[:3, :3]
    t = reg_p2p.transformation[:3, 3][:, None]
    
    return R, t

def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input.rstrip('\n')

    input_file = open(os.path.join(args.data_path, 'models', class_input, 'points.xyz'))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

for now in range(0, 2949):
    img = Image.open(os.path.join(args.data_path, testlist[now]+'-color.png'))
    depth = np.array(Image.open(os.path.join(args.data_path, testlist[now]+'-depth.png')))
    posecnn_meta = scio.loadmat('./{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    
    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    lst = posecnn_rois[:, 1:2].flatten()
    my_result_wo_refine = []
    my_result = []
    

    my_batch = next(iter(test_loader))
    for idx in range(len(lst)):
        itemid = lst[idx]
        try:
            rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])


            batch = {
            'rgb': torch.tensor([0]),
            'depth': torch.tensor([0]), # depth of whole image
            'rgb_masked': img_masked,
            'pts_sensor_cam': cloud, # observed point cloud of object in camera frame
            'pose': torch.tensor([0]),
            'mask_in_bbox': choose,
            'model_points': torch.tensor([0]),   # 3d points of object model
            'raw_model_points': torch.tensor([0]),
            'pts_model_cam': torch.tensor([0]), # 3d points of object model in camera frame
            'model_index': index,
            'bbox': torch.tensor([0]),
            'gt_3d_vector_field': my_batch['gt_3d_vector_field'],
            'gt_2d_vector_field': torch.tensor([0]),
            'pts_farthest_model': my_batch['pts_farthest_model'], 
            'pts_farthest_cam': torch.tensor([0])
        }


            model.set_input(batch)
            pred_r, pred_t= model.forward()


            pred_r = pred_r.squeeze().cpu().numpy()
            pred_t = pred_t.cpu().numpy()
            pred_R = quaternion_matrix(pred_r)[:3, :3]
            t = np.reshape(pred_t, (3,1))

            init_pred = np.append(pred_r, pred_t)
            my_result_wo_refine.append(init_pred.tolist())

            # refine
            icp_R, icp_t = icp_refine(pred_R, t, my_batch['model_points'], cloud)
            icp_quat = quaternion_from_matrix(icp_R)
            icp_pred =np.append(icp_quat, icp_t)
            my_result.append(icp_pred.tolist())
            
        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

    scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now), {'poses':my_result_wo_refine})
    scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now), {'poses':my_result})
    print("Finish No.{0} keyframe".format(now))