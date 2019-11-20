import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

from .base_model import BaseModel
from . import networks
from models.networks import define_seg_net
import models.loss as Criterion
from models.network_utils import get_pose_pred, get_dir_pred_distribution, \
    get_farthestpts_pred_distribution
from utils.tf_torch import combine_two_poses



class SegModel(BaseModel):
    """ This class implements the model, 
          for further aligning two point clouds given initial pose.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        args:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        returns:
            the modified parser.

        """
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')

        args = parser.parse_args()
        
        if not is_train:
            return parser
        if args.lr_policy == 'step':
            parser.add_argument('--lr_decay_iters', type=int, default=80, 
                help='multiply by a gamma every lr_decay_iters iterations')
            parser.add_argument('--lr_decay_gamma', type=float, default=0.5, 
                help='multiply by lr_decay_gamma every lr_decay_iters iterations')
        elif args.lr_policy == 'linear':
            parser.add_argument('--niter', type=int, default=100, 
                help='# of iter at starting learning rate')
            parser.add_argument('--niter_decay', type=int, default=100, 
                help='# of iter to linearly decay learning rate to zero')
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
    
        self.model_names = ['M']
        
        if opt.dataset == 'linemod':
            npt = opt.num_points
            nobj = 13
            self.sym_list = [7, 8]
        elif opt.dataset == 'ycb':
            npt = opt.num_points
            nobj = 21
            self.sym_list = [12, 15, 18, 19, 20]
        else:
            raise NotImplementedError('dataset [%s] is not implemented', opt.dataset)
        
        self.netM = define_seg_net(net=opt.net, npt=npt, nobj=nobj, 
            pt_emb=opt.point_emb, device=self.device, 
            init_type='normal', init_gain=0.02, gpu_ids=[0])
        
        if self.isTrain:
            self.netM.train()
            class_weights = None
            self.alpha = 1.0
            self.seg_criterion = nn.NLLLoss(weight=class_weights)
            self.cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)

            self.optimizer = torch.optim.Adam(self.netM.parameters(), 
                lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer)
        else:
            self.netM.eval()


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        args:
            input (dict): include the data itself and its metadata information.
                - rgb: (bs, 3, H, W)
                - seg_mask: (bs, H, W)
                - gt_cls: (bs, num_class)
        """
        self.rgb = input['rgb'].to(self.device)
        self.seg_mask = input['seg_mask'].to(self.device)
        self.model_index = input['model_index']

        # self.gt_cls = input['gt_cls'].to(self.device)   

    def forward(self):
        self.out, self.out_cls = self.netM(self.rgb)
        self.out = self.out[:, self.model_index[0][0].numpy(), :, :]
        return self.out, self.out_cls
 
    def backward(self):
        seg_loss = self.seg_criterion(self.out, self.seg_mask)
        # cls_loss = self.cls_criterion(self.out_cls, self.gt_cls)
        self.loss_pred = seg_loss #+ self.alpha * cls_loss
        self.loss_pred.backward()

    def optimize_parameters(self, step_optimizer):
        self.forward()                  
        self.backward()
        if step_optimizer:
            self.optimizer.step()     
            self.optimizer.zero_grad()
