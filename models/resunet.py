import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from models.common import get_norm

from models.residual_block import get_block


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class STN3d(nn.Module):
    r"""Given a sparse tensor, generate a 3x3 transformation matrix per
    instance.
    """
    CONV_CHANNELS = [64, 128, 1024, 512, 256]
    # CONV_CHANNELS = [64, 128, 32, 16, 32]
    FC_CHANNELS = [512, 256]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self, in_channels=3, out_channels=3, D=3):
        super(STN3d, self).__init__()

        NORM_TYPE = 'BN'
        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS

        self.conv1 = ME.MinkowskiConvolution(
            in_channels, c[0], kernel_size=k[0], stride=s[0], has_bias=False, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            c[0],
            c[1],
            kernel_size=k[1],
            stride=s[1],
            has_bias=False,
            dimension=3)
        self.conv3 = ME.MinkowskiConvolution(
            c[1],
            c[2],
            kernel_size=k[2],
            stride=s[2],
            has_bias=False,
            dimension=3)

        # Use the kernelsize 1 convolution for linear layers. If kernel size ==
        # 1, minkowski engine internally uses a linear function.
        self.fc4 = ME.MinkowskiConvolution(
            c[2], c[3], kernel_size=1, has_bias=False, dimension=3)
        self.fc5 = ME.MinkowskiConvolution(
            c[3], c[4], kernel_size=1, has_bias=False, dimension=3)
        self.num_objs = 13
        self.fc6 = ME.MinkowskiConvolution(
            c[4], out_channels, kernel_size=1, has_bias=True, dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.avgpool = ME.MinkowskiGlobalPooling(dimension=D)
        self.broadcast = ME.MinkowskiBroadcast(dimension=D)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        #x = self.avgpool(x)

        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        x = self.fc6(x)

        return x 


class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    #out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    #out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    #out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    #out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)

    out = self.conv4_tr(out)
    #out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat((out_s4_tr, out_s4))

    out = self.conv3_tr(out)
    #out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat((out_s2_tr, out_s2))

    out = self.conv2_tr(out)
    #out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat((out_s1_tr, out_s1))
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out

class FusionFeature(nn.Module):
    def __init__(self, nc_pt=32):
        '''Fuse features of rgb and point cloud
        args:
            num_points
            nc_rgb: number of channels of rgb feature
            nc_pt: number of channels of point cloud feature
        '''
        super(FusionFeature, self).__init__()
  
        self.pt_conv1 = torch.nn.Conv1d(nc_pt, 64, 1)
        self.pt_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(128, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 512, 1)
        
    def forward(self, emb_pt):
        num_pts = emb_pt.shape[2]

        emb_pt = F.relu(self.pt_conv1(emb_pt))
        fusedfeat_1 = emb_pt

        emb_pt = F.relu(self.pt_conv2(emb_pt))
        fusedfeat_2 = emb_pt

        fusedfeat_3 = F.relu(self.conv5(fusedfeat_2))
        fusedfeat_3 = F.relu(self.conv6(fusedfeat_3))
        self.ap1 = torch.nn.AvgPool1d(num_pts)
        # print('aa what', fusedfeat_3.shape)
        fusedfeat_3 = self.ap1(fusedfeat_3)
        # print('what', fusedfeat_3.shape)
        fusedfeat_3 = fusedfeat_3.view(-1, 512, 1).repeat(1, 1, num_pts)
        # print('fused', fusedfeat_1.shape, fusedfeat_2.shape, fusedfeat_3.shape)
        return torch.cat([fusedfeat_1, fusedfeat_2, fusedfeat_3], 1) #128 + 256 + 1024


class GeoPoseNet(ME.MinkowskiNetwork):
    def __init__(self, num_points, num_objs, D):
        super(GeoPoseNet, self).__init__(D)

        self.num_points = num_points
        self.num_objs = num_objs
        self.device = [0] 
        self.num_farthest_pts = 8

        self.emb_pt =ResUNetBN2C(in_channels=1,
              out_channels=3*self.num_objs*self.num_farthest_pts,
              bn_momentum=0.1,
              normalize_feature=False,
              conv1_kernel_size=7,
              D=D)
       
        # self.emb_pt.load_state_dict(torch.load('../checkpoints/two_pcd/checkpoint_FCGF.pth')['state_dict'])
        # for param in self.emb_pt.parameters():
        #    param.requires_grad = False
        
        self.stn = STN3d(in_channels=32, out_channels=3*self.num_objs*self.num_farthest_pts, D=D)
        # self.fusion_future = FusionFeature(nc_pt=32)

        # self.conv1 = torch.nn.Conv1d(1408//2, 640, 1)
        # self.conv1 = torch.nn.Conv1d(1408//2, 256, 1)
        # self.conv2 = torch.nn.Conv1d(256, 128, 1)
        # self.conv3 = torch.nn.Conv1d(128, 3*self.num_farthest_pts, 1) #translation
       
    def forward(self, sinput):
        '''
        args:
            pts_sensor_cam: (bs, num_pts, 3), point cloud in camera frame
            obj: (bs, 1), object index
        '''
        num_pts, _ = sinput.shape
        emb_pt = self.emb_pt(sinput) # (num_pts, 32)
        # emb_pt = emb_pt.contiguous().transpose(1, 0).contiguous()
        # print(emb_pt.shape)
        # emb_pt = self.fusion_future(emb_pt[None, :])
        # out = F.relu(self.conv1(emb_pt))
        # out = F.relu(self.conv2(out))
        # out = self.conv3(out)
        # print(out.shape)
        return emb_pt#self.stn(emb_pt) 
        
        # emb_pt = emb_pt.F.transpose(1,0)[None, :]
        # dirs = self.conv(emb_pt)
        # pred_dirs = dirs.view(self.num_points, self.num_farthest_pts, 3)
        # out = dirs.transpose(1, 0).view(self.num_objs, 3*self.num_farthest_pts, self.num_points)
        
        # b = 0
        # pred_dirs = torch.index_select(out, 0, obj[b])
        # pred_dirs = pred_dirs.contiguous().transpose(2, 1).contiguous()
        # pred_dirs = pred_dirs.view(self.num_points, self.num_farthest_pts, 3)


        #emb_pt = emb_pt.F.transpose(1, 0).contiguous()[None, :]
        #bs = 1

        #emb_fused = self.emb_fusion(emb_pt)

        #dirs = F.relu(self.conv1_d(emb_fused))
        #dirs = F.relu(self.conv2_d(dirs))
        #dirs = F.relu(self.conv3_d(dirs))
        #dirs = self.conv4_d(dirs)
        #dirs = dirs.view(bs, self.num_objs, 3*self.num_farthest_pts, self.num_points)
        #
        #b = 0
        #pred_dirs = torch.index_select(dirs[b], 0, torch.tensor(obj).cuda())
        pred_dirs = emb_pt.F

        #pred_dirs = pred_dirs.contiguous().transpose(2, 1).contiguous()
       

        pred_dirs = pred_dirs.view(1, self.num_points, self.num_farthest_pts, 3)
        
        return pred_dirs

class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'
