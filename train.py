# ha
import time
import os
import numpy as np
import open3d as o3d
import torch.utils.data
import logging

from models import create_model
from options.train_options import TrainOptions
from utils.custom_logger import Logger
from dataloader.make_dataloader import make_dataloader
from utils.metrics import AverageMeter, evaluate_hit_ratio
from utils.eval_utils import compute_adds_metric, is_correct_pred

args = TrainOptions().parse()
visualizer = Logger(args)
model = create_model(args) 
model.setup(args) 

if args.select_obj is not None:
    select_obj = [int(item) for item in args.select_obj.split(',')]
else:
    select_obj=None
train_loader = make_dataloader(args.dataset, args.data_path, args.phase,
    args.batch_size, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=True, select_obj=select_obj)
val_loader = make_dataloader(args.dataset, args.data_path, 'test',
    1, args.voxel_size, args.num_points, num_threads=args.workers,
    shuffle=False, select_obj=select_obj)
val_set = val_loader.dataset

logging.getLogger().setLevel(logging.INFO)

def train(epoch):
    model.set_phase('train')
    iter_data_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if not batch:
            continue 
        iter_start_time = time.time()
        if i % args.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        model.set_input(batch)       
        if (i+1) % args.step_freq == 0 and i != 0:
            step = True
        else:
            step = False
        model.optimize_parameters(step)

        with torch.no_grad():
            if i % args.print_freq == 0:    
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / args.batch_size
                visualizer.print_current_losses(
                    epoch, i, losses, t_comp, len(train_loader))

            # if i % args.display_freq == 0:   
            #     rgb_images = [model.rgb]
            #     depth_images = [batch['seg_mask'], model.out]
            #     visualizer.save_and_display_images(
            #         'train', epoch, i, len(train_loader), rgb_images, depth_images)

            iter_data_time = time.time()

    model.update_learning_rate()

# only use in debug mode
def validate(epoch):
    model.set_phase('val')
    val_distance = 0
    len_loader = len(val_loader)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, adds_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    num_models = val_set.get_num_of_models()
    total_instance_cnt = [0 for i in range(num_models)]
    success_cnt = [0 for i in range(num_models)]
    for i, batch in enumerate(val_loader):

        if not batch:
            continue 
        if i % 10 != 0:
           continue

        with torch.no_grad():
            model.set_input(batch)  
            T_est = model.forward() 
            T_gt = batch['T_gt'][0].numpy()

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            # print(T_est, 'gt', T_gt)
            rte_meter.update(rte)

            rre = np.arccos((np.trace(T_est[:3, :3].T @ T_gt[:3, :3]) - 1) / 2)
            
            if not np.isnan(rre):
                rre_meter.update(rre)
            tl1loss = np.mean(np.abs(T_est[:3, 3] - T_gt[:3, 3]))
            loss_meter.update(tl1loss)

            # hit_ratio = evaluate_hit_ratio(
            #     model.pred_dirs.cpu().numpy().reshape((-1, 3)), 
            #     batch['gt_dir_vecs_c'].cpu().numpy().reshape((-1, 3)), 
            #     thresh=args.hit_ratio_thresh)
            # hit_ratio_meter.update(hit_ratio)

            # feat_match_ratio.update(hit_ratio > 0.05)

            obj = batch['model'][0] # currently val only supports batch size = 1 
            model_points = obj.get_model_points()
            diameter = obj.get_model_diameter()
            model_index = obj.get_index()
            is_sym = obj.is_symmetric()

            distance = compute_adds_metric(T_est[:3, :3], T_est[:3, 3][:, None], 
                T_gt[:3, :3], T_gt[:3, 3][:, None],
                model_points, is_sym) #currently val only supports batch size=1
            adds_meter.update(distance)
            if is_correct_pred(distance, diameter): 
                success_cnt[model_index] += 1
            total_instance_cnt[model_index] += 1

            num_data += 1
            torch.cuda.empty_cache()

            if i % 3000 == 0 and i > 0:
                logging.info(' '.join([
                    f"Validation iter {num_data} / {len_loader} : ",
                    f"T L1: {loss_meter.avg:.3f},",
                    f"RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                    # f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
                ]))

    logging.info(' '.join([
        f"Avg Distance: {adds_meter.avg:.3f},",
        f"RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    
    for i in range(num_models):
        if total_instance_cnt[i] != 0:
            model_id = val_set.obj_ids[i]
            model_name = val_set.obj_dics[model_id]
            print('Model {0} success rate: {1}'.format(
                model_name, float(success_cnt[i]) / total_instance_cnt[i]))
    print('Overal success rate: {0}'.format(
        float(sum(success_cnt)) / sum(total_instance_cnt)))
    return adds_meter.avg
          
def main():
    best_dis = float("inf")
    for epoch in range(args.epoch_start, args.epoch_end):
        if epoch % args.valid_freq == 0:
            mean_dis = validate(epoch)
            if mean_dis < best_dis: 
                save_suffix = os.path.join(
                    args.checkpoints_dir, args.name, 'val_best')
                model.save_networks(save_suffix)

        train(epoch)

        save_suffix = os.path.join(
           args.checkpoints_dir, args.name, 'latest')
        model.save_networks(save_suffix)

if __name__ == '__main__':
    main()
