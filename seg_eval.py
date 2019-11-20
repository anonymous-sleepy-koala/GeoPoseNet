import os
import numpy as np
import torch.utils.data
from tqdm import tqdm
import time
from PIL import Image

from options.test_options import TestOptions
from models import create_model
from utils.visualizer import Visualizer
import utils.open3d_utils as o3d_utils
import utils.eval_utils as eval_utils
import utils.tf_numpy as tf_numpy



args = TestOptions().parse()
visualizer = Visualizer(args)
model = create_model(args) 
model.setup(args) 

if args.dataset == 'linemod':
    from dataloader.linemod_dataloader import LineMOD
    test_set = LineMOD(args.data_path, 'test', args.num_points)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.workers, sampler=None)
elif args.dataset == 'ycb':
    from dataloader.ycb_dataloader import YCBVideo
    test_set = YCBVideo(args.data_path, 'test', False, args.num_points)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.workers, sampler=None)
else:
    raise NotImplementedError("dataset not implemented!")
                
def test():
    print('running segmentation model, evaluate results...')
    assert args.batch_size == 1, "batch size can only be 1"
    num_classes = test_set.get_num_of_models()
    iou_res = eval_utils.IoU(num_classes)
    for i, batch in enumerate(tqdm(test_loader)):
        if not batch:
            continue
        # if i > 1000:
        #     break
       
        model_index = batch['model_index'].numpy()[0][0]
        model_id = test_set.obj_ids[model_index]
        
        label_pred = np.zeros((args.image_height, args.image_width))
        label_posecnn = batch['posecnn_mask'].numpy()[0]
        with torch.no_grad():
            model.set_input(batch) 
            out, out_cls = model.forward()  
            res = out[0].max(0)[1]
            res = res.cpu().numpy()
            label_pred[res == 1] = 255
            
            iou_res.add(label_pred, batch['seg_mask'].cpu().numpy()[0], model_index)
            if args.save_seg:
                img = Image.fromarray(label_pred.astype(np.uint8))

                gt_mask_path =test_set.paths['mask'][i]
                partial_path = gt_mask_path.split('/')[-3:]
                val_output_dir = os.path.join(
                    args.checkpoints_dir, args.name, 'images_val', partial_path[-3])
                if not os.path.exists(val_output_dir):
                    os.makedirs(val_output_dir)
                img.save(os.path.join(val_output_dir, partial_path[2]))

                # label_gt = batch['seg_mask'][0].cpu().numpy()*255
                # img = Image.fromarray(label_gt.astype(np.uint8))
                # file = os.path.join(
                #     args.checkpoints_dir, args.name, 'images_val', '%05d_gt.png' % i)
                # img.save(file)
    for i in range(num_classes):
        model_id = test_set.obj_ids[i]
        model_name = test_set.obj_dics[model_id]
        print('Model {0} avg. IoU: {1}'.format(
            model_name, iou_res.res[i, 2]))
    res_all = np.sum(iou_res.res, 0)
    print('Overal IoU: {0}'.format(res_all[0] / res_all[1]))

    
def main():
    test()
    
if __name__ == '__main__':
    main()
