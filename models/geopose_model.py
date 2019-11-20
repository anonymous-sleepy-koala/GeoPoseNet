import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import MinkowskiEngine as ME

from .base_model import BaseModel
from . import networks
from models.networks import define_geopose_net
import models.loss as Criterion
from models.network_utils import get_pose_pred_batch
from utils.tf_torch import quaternion2matrix
from utils.metrics import AverageMeter

import time


class GeoPoseModel(BaseModel):
    """ This class implements the Point Cloud model (PCDModel), 
          for estimating the pose of input 2.5D point cloud.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        # if is_train:
            # parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        args = parser.parse_args()
        
        if not is_train:
            return parser
        if args.lr_policy == 'step':
            parser.add_argument('--lr_decay_iters', type=int, default=100, 
                help='multiply by a gamma every lr_decay_iters iterations')
            parser.add_argument('--lr_decay_gamma', type=float, default=0.5, 
                help='multiply by lr_decay_gamma every lr_decay_iters iterations')
        elif args.lr_policy == 'linear':
            parser.add_argument('--niter', type=int, default=100, 
                help='# of iter at starting learning rate')
            parser.add_argument('--niter_decay', type=int, default=100, 
                help='# of iter to linearly decay learning rate to zero')
        elif args.lr_policy == 'exponential':
            pass
        else:
            raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return parser

    def __init__(self, opt):
        """Initialize the class.

        args:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        self.loss_names = ['pred']
        # specify the images you want to save/display. 
        self.visual_names = []        
        # define networks
        if self.opt.run_refine_net:
            self.model_names = ['M', 'R'] # MatchingNet, RefiningNet
        else:
            self.model_names = ['M']
        
        self.num_farthest_pts = 8
        if opt.dataset == 'LineMOD':
            npt = opt.num_points
            self.nobj = 13
            self.sym_list = [7, 8]
        elif opt.dataset == 'ycb':
            npt = opt.num_points
            self.nobj = 21
            self.sym_list = [12, 15, 18, 19, 20]
        elif opt.dataset == 'LineMODOcclusion':
            npt = opt.num_points
            self.nobj = 13
            self.sym_list = [7, 8]
        else:
            raise NotImplementedError('dataset [%s] is not implemented', opt.dataset)
        
        self.netM = define_geopose_net(npt=npt, 
            nobj=self.nobj, 
            device=self.device, 
            init_type='normal', 
            init_gain=0.02, 
            gpu_ids=[0])
        
        if self.isTrain:
            self.netM.train()
            self.criterionM = nn.MSELoss() #nn.L1Loss()#Criterion.ADDSLoss(self.sym_list) #nn.L1Loss()
            self.optimizer = torch.optim.Adam(self.netM.parameters(), 
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizer = torch.optim.SGD(self.netM.parameters(), 
            #     lr=1e-1, momentum=0.8, weight_decay=1e-4)

            self.optimizers.append(self.optimizer)
            self.current_phase = 'train'
        else:
            self.netM.eval()
            self.current_phase = 'val'
        self.forward_timer = AverageMeter()
        self.ls_timer = AverageMeter()
        self.svd_timer = AverageMeter()



    def set_phase(self, phase):
        self.current_phase = phase
        if phase == 'train':
            self.netM.train()
        else:
            self.netM.eval()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.input = input
        
        # input data of sensors
        self.xyz_s = self.input['xyz_s'].to(self.device)
        self.center_s = self.input['center_s'].to(self.device)
        self.len_s = self.input['len_s']

        self.gt_dir_vecs_c = self.input['gt_dir_vecs_c'].to(self.device)
        self.pts_farthest_m = self.input['pts_farthest_m'][0].to(self.device)
        

        obj = self.input['model'][0] # currently val only supports batch size = 1 
        pts_model = obj.get_model_points()
        self.model_points = torch.from_numpy(pts_model[None, :]).to(self.device)
        self.model_index = obj.get_index()
        self.pose_gt = self.input['T_gt'][0].to(self.device)
        # for item in input:
        #     for k, v in input.items():
        #         if v is not None:
        #             self.input[k] = v.to(self.device)

    def forward(self): 
        start = time.time() 
        sinput_s = ME.SparseTensor(
            self.input['feats_s'], coords=self.input['coords_s']).to(self.device)
        self.pred_dirs = self.netM(sinput_s).F # expect output: (num_pts, 13*8*3)
        #print('1', self.input['coords_s'].shape, sinput_s.shape, self.pred_dirs.shape)
        
        self.pred_dirs = self.pred_dirs.view(-1, self.nobj, self.num_farthest_pts, 3)
        num_pts = self.pred_dirs.shape[0]
        self.output = torch.zeros(num_pts, self.num_farthest_pts, 3).to(self.device)
        s = 0
        for i, num in enumerate(self.len_s):
            num = num[0]
            self.output[s:s+num, :, :] = torch.index_select( \
                self.pred_dirs[s: s+num, :], 1, torch.tensor(self.model_index).to(self.device)).squeeze(1)
            s += num
        
        self.pred_dirs = self.output.view(-1, 8, 3)
        end = time.time()
        self.forward_timer.update(end-start)
        # self.pred_dirs = self.gt_dir_vecs_c
        #print(self.pred_dirs.shape, self.gt_dir_vecs_c.shape)

        #norm = torch.norm(self.pred_dirs, dim=2, keepdim=True).expand(-1, -1, 3)
        #self.pred_dirs = self.pred_dirs.div(norm)

        if self.current_phase == 'val':
            # assume single batch
            pred_rot, pred_t, t1, t2 = get_pose_pred_batch(self.xyz_s, 
                self.pred_dirs, self.pts_farthest_m, self.input['len_s'])
            self.ls_timer.update(t1)
            self.svd_timer.update(t2)
            #print(pred_rot[0].shape, pred_t[0].shape)
            pred_T = np.concatenate((pred_rot[0], pred_t[0]), axis=1)
            pred_T = np.concatenate((pred_T, np.array([[0, 0, 0, 1]])), axis=0)
            # np.savetxt('../pose_results/' + str(self.index) + '.txt', pred_T)
            return pred_T

    def backward(self):
        # TODO: normalize pred_dirs
        self.loss_pred = self.criterionM(self.pred_dirs, self.gt_dir_vecs_c)
        # self.loss_pred = self.criterionM(self.pred_rot, self.pred_trans, 
        #     self.model_points, self.pose_gt, self.model_index, self.device)
        self.loss_pred.backward()

    def optimize_parameters(self, step_optimizer):
        self.forward()                  
        self.backward()
        if step_optimizer:
            self.optimizer.step()     
            self.optimizer.zero_grad()
