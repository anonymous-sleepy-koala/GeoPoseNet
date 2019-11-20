#!/usr/bin/python
import sys
sys.path.append('..')
import argparse
import os
from dataloader.linemod_dataloader import LineMOD
from dataloader.ycb_dataloader import YCBVideo



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
        type=str,
        default='linemod',
        help='dataset option: [linemod | ycb]')
    parser.add_argument("--data_path",
        type=str,
        default= '../data/Linemod_preprocessed',
        help="path to dataset")
    parser.add_argument('--visualize',
        action='store_true',
        help='visualize point cloud and found farthest_point')
    parser.add_argument('--write_to_file',
        action='store_true',
        help='save computed results to files')
    args = parser.parse_args()

    if args.dataset == 'linemod':
        dataset = LineMOD(args.data_path, 'train', num_points=1000)
    elif args.dataset == 'ycb':
        dataset = YCBVideo(args.data_path, 'train', num_points=1000)
    else:
        raise NotImplementedError("dataset nor implemented!")

    
    generate_num = [4, 8, 12, 16, 20]
    for key, model in dataset.models.items():
        for num in generate_num:
            model.compute_farthest_surface_point(num_pt=num, 
                write_to_file=args.write_to_file)

    # visualize results for verification
    if args.visualize:
        dataset.models[11].show_model(8)
            