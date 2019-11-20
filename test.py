import numpy as np
from tqdm import tqdm
import time
import logging

import utils.open3d_utils as o3d_utils
from models import create_model
from utils.custom_logger import Logger
import utils.eval_utils as eval_utils
import utils.tf_numpy as tf_numpy
from utils.metrics import AverageMeter
from dataloader.make_dataloader import make_dataloader
from options.test_options import TestOptions
import torch.utils.data


args = TestOptions().parse()
visualizer = Logger(args)
model = create_model(args) 
model.setup(args) 

if args.select_obj is not None:
    select_obj = [int(item) for item in args.select_obj.split(',')]
else:
    select_obj=None
test_loader = make_dataloader(args.dataset, args.data_path, 'test',
    1, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=False, select_obj=select_obj)
test_set = test_loader.dataset
  
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')
def test():
    print()
    print('Running GeoPoseNet, predicting poses and evaluating results...')
    assert args.batch_size == 1, "batch size can only be 1"
    num_models = test_set.get_num_of_models()
    total_instance_cnt = [0 for i in range(num_models)]
    success_cnt = [0 for i in range(num_models)]
    
    quat_distance_sum = 0
    quat_angle_sum = 0
    x_error = 0
    y_error = 0
    z_error = 0

    loss_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
    t_x_meter, t_y_meter, t_z_meter = AverageMeter(), AverageMeter(), AverageMeter()
    adds_meter = AverageMeter()
    start = time.time()
    runtime_timer = AverageMeter()
    #slots = {0.99:0, 0.95:0, 0.9:0, 0.8:0, 0.5:0}
    #slots_far = {0.007:0, 0.002:0, 0.005:0, 0.01:0, 0.02:0}
    start = time.time()
    for i, batch in enumerate(tqdm(test_loader)):
        start = time.time()
        if not batch:
            continue
        # if i > 5000: 
        #     break 
        with torch.no_grad():
            if True:
                model.set_input(batch) 
                model_index = model.model_index
                model_id = test_set.obj_ids[model_index]
                pose_gt = model.pose_gt.cpu().numpy()
                diameter = test_set.models[model_id].get_model_diameter()

                obj = batch['model'][0] # currently val only supports batch size = 1 
                model_points = obj.get_model_points()
                #best_quat, best_t, res, res_far = model.forward()  
                pose_pred = model.forward()  
                                
                rot_pred = pose_pred[:3, :3]
                t_pred = pose_pred[:3, 3]

                quat_pred = tf_numpy.quaternion_from_matrix(rot_pred)
                t_pred = t_pred[:, None]
            else:
                voxel_size = 0.01
                # pose_pred = o3d_utils.fgr(batch['model_points'].squeeze().cpu().numpy(), 
                #     batch['pts_sensor_cam'].squeeze().cpu().numpy(), voxel_size)
                pose_pred = o3d_utils.ransac_global_registration(
                    model_points.numpy(), 
                    model.xyz_s.squeeze().cpu().numpy(),
                    voxel_size)
                R = pose_pred[:3, :3]
                t = pose_pred[:3, 3][:, None]
                best_quat = tf_numpy.quaternion_from_matrix(R)
            if args.do_icp_refine:
                rot_refined, t_refined = o3d_utils.icp_refine(rot_pred, t_pred, 
                    model_points, 
                    model.xyz_s.cpu().numpy())
                quat_refined = tf_numpy.quaternion_from_matrix(rot_refined)

                rot_pred = rot_refined
                t_pred = t_refined
                quat_pred = quat_refined
            
            # Relative Translation Error (RTE)
            rte = np.linalg.norm(t_pred - pose_gt[:3, 3][:, None])
            rte_meter.update(rte)
            # Relative Rotation Error (RRE)
            rre = np.arccos((np.trace(rot_pred.T @ pose_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)
            # Ralative Translation Error in x, y, z direction
            q_distance, quat_angle, x_offset, y_offset, z_offset = \
                eval_utils.compute_error(quat_pred, t_pred, 
                    tf_numpy.quaternion_from_matrix(pose_gt[:3, :3]),\
                    pose_gt[:3, 3][:, None])
            t_x_meter.update(x_offset)
            t_y_meter.update(y_offset)
            t_z_meter.update(z_offset)

            # Average closest point distance ( ADD(S) )
            distance = eval_utils.compute_adds_metric(
                rot_pred, t_pred, 
                pose_gt[:3, :3], pose_gt[:3, 3][:, None],
                model_points,
                test_set.is_symmetric_obj(model_index))
            adds_meter.update(distance)
            # print(distance)
            threshold = 1 if args.dataset == 'ycb' else 0.1
            if eval_utils.is_correct_pred(distance, diameter, threshold):
                success_cnt[model_index] += 1
            total_instance_cnt[model_index] += 1
            end = time.time()
            runtime_timer.update(end-start)
            start = end

    # result summary
    print()
    print('time elapse:', time.time() - start)
    print()
    logging.info(' '.join([
        f"RTE: {rte_meter.avg:.3f}, ",
        f"RTE x: {t_x_meter.avg:.5f}, y: {t_y_meter.avg:.5f}, z: {t_z_meter.avg:.5f}"
    ]))
    logging.info(' '.join([
        f"RRE: {rre_meter.avg:.3f}"
    ]))
    logging.info(' '.join([
        f"ADD: {adds_meter.avg:.5f}"
    ]))
    
    print()
    print('ADD(S) Metrics')
    for i in range(num_models):
        if total_instance_cnt[i] != 0:
            model_id = test_set.obj_ids[i]
            model_name = test_set.obj_dics[model_id]
            print('Model {0} success rate: {1}'.format(
                model_name, float(success_cnt[i]) / total_instance_cnt[i]))
    print('Overal success rate: {0}'.format(
        float(sum(success_cnt)) / sum(total_instance_cnt)))
    print('dataloading time', test_loader.dataset.timer.avg)
    print('forward time', model.forward_timer.avg)
    print('ls time', model.ls_timer.avg)
    print('svd time', model.svd_timer.avg)
    print('overal time', runtime_timer.avg)



    
def main():
    test()
    
if __name__ == '__main__':
    main()
