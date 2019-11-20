import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# from options.train_options import TrainOptions
from dataloader.linemod_dataset import LineMOD
from dataloader.linemod_occlusion_dataset import LineMODOcclusion
import dataloader.data_utils as dutils
import torch



def collate_pair_fn(list_data):
    r"""
    0: model points
    1: sensor points
    """
    xyz0, coords0, feats0, center0, \
    xyz1, coords1, feats1, center1, \
    models, \
        pts_farthest_m, pts_farthest_c, gt_dir_vecs_c, trans = list(zip(*list_data))
  
    xyz_batch0, coords_batch0, feats_batch0, center_batch0 = [], [], [], []
    xyz_batch1, coords_batch1, feats_batch1, center_batch1 = [], [], [], []

    model_batch = []
    pts_farthest_m_batch0, pts_farthest_c_batch0 = [], [] 
    gt_dir_vecs_batch0, trans_batch = [], []
    len_batch0, len_batch1 = [], []

    batch_id = 0
    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]

        xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
        xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))

        coords_batch0.append( torch.cat((torch.from_numpy(
                coords0[batch_id]).int(), torch.ones(N0, 1).int() * batch_id), 1))
        coords_batch1.append( torch.cat((torch.from_numpy(
                coords1[batch_id]).int(), torch.ones(N1, 1).int() * batch_id), 1))

        feats_batch0.append(torch.from_numpy(feats0[batch_id]))
        feats_batch1.append(torch.from_numpy(feats1[batch_id]))

        center_batch0.append(torch.from_numpy(center0[batch_id])[None, -1])
        center_batch1.append(torch.from_numpy(center1[batch_id])[None, -1])

        len_batch0.append([N0])
        len_batch1.append([N1])

        model_batch.append(models[batch_id])

        pts_farthest_m_batch0.append(
            torch.from_numpy(pts_farthest_m[batch_id]).float() )
        pts_farthest_c_batch0.append(
            torch.from_numpy(pts_farthest_c[batch_id]).float() )

        trans_batch.append(torch.from_numpy(trans[batch_id]).float())
        
        gt_dir_vecs_batch0.append(
            torch.from_numpy(gt_dir_vecs_c[batch_id]) )

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    coords_batch0 = torch.cat(coords_batch0, 0).int()
    feats_batch0 = torch.cat(feats_batch0, 0).float()
    center_batch0 = torch.cat(center_batch0, 0).float()

    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    coords_batch1 = torch.cat(coords_batch1, 0).int()
    feats_batch1 = torch.cat(feats_batch1, 0).float()
    center_batch1 = torch.cat(center_batch1, 0).float()
 
    gt_dir_vecs_batch0 = torch.cat(gt_dir_vecs_batch0, 0).float()

    return {
      'xyz_m': xyz_batch0,
      'coords_m': coords_batch0,
      'feats_m': feats_batch0,
      'center_m': center_batch0,
      'xyz_s': xyz_batch1,
      'coords_s': coords_batch1,
      'feats_s': feats_batch1,
      'center_s': center_batch1,
      'len_m': len_batch0,
      'len_s': len_batch1,
      'model': model_batch,
      'pts_farthest_m': pts_farthest_m_batch0,
      'pts_farthest_c': pts_farthest_c_batch0,
      'gt_dir_vecs_c': gt_dir_vecs_batch0,
      'T_gt': trans_batch,
    }
    

ALL_DATASETS = [LineMOD, LineMODOcclusion]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}

def make_dataloader(dataset, data_path, phase, batch_size, voxel_size, num_points, num_threads=0, shuffle=None, select_obj=None):
    assert phase in ['train', 'test']
    if shuffle is None:
        shuffle = phase != 'test'

    if dataset not in dataset_str_mapping.keys():
        logging.error(f'Dataset {dataset}, does not exists in ' +
                      ', '.join(dataset_str_mapping.keys()))

    Dataset = dataset_str_mapping[dataset]
    dset = Dataset(data_path, phase, voxel_size, num_points, select_obj=select_obj) 


    loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      drop_last=True)

    return loader



if __name__ == '__main__':
    voxel_size=0.002
    loader = make_dataloader(dataset='LineMOD', 
      data_path='../data/Linemod_preprocessed', 
      phase='test', batch_size=1, voxel_size=voxel_size, 
      num_points=5000, num_threads=1, shuffle=False)
    data_loader_iter = loader.__iter__()
    input_dict = data_loader_iter.next()


    xyz0 = input_dict['xyz0'].numpy()
    xyz1 = input_dict['xyz1'].numpy()
    trans = input_dict['T_gt'][0].numpy()

    xyz0_transformed = dutils.apply_transform(xyz0, trans)

    pcd0 = dutils.make_open3d_point_cloud(xyz0_transformed)
    pcd1 = dutils.make_open3d_point_cloud(xyz1)
    # # find the density of point cloud

    # # find the overlap of two point clouds
    
    ratio = dutils.compute_overlap_ratio(pcd0, pcd1, voxel_size, 
      match_thresh_ratio=2.5)
    print(ratio)
    # Make point clouds using voxelized points
    # pcd = dutils.make_open3d_point_cloud(xyz)
    # Select features and points using the returned voxelized indices
    # pcd.colors = o3d.utility.Vector3dVector(color[sel])
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[sel])

