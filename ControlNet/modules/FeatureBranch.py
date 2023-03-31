import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ResNeXt.resnext_50_32x4d import ResNext
from modules.util import ResBlock3d, UpBlock2d, DownBlock2d, create_sparse_motions, EqualLinear

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    zero_module,
)
from ldm.modules.encoders.xf import Transformer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Feature_Branch(nn.Module):

    def __init__(self, 
                 reshape_channel=None, 
                 reshape_depth=None, 
                 num_resblocks=None, 
                 upsample_channels=None, 
                 downsample_channels=None,  
                 warp_feature = False
                 ):
        super().__init__() 
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth
        self.warp_feature = warp_feature

        # ResNeXt for global extractor
        self.global_extractor = ResNext().network()

        self.mapper = Transformer(
                1,
                1024,
                5,
                8,
            )
        
        # global residual (add to id feature)
        self.global_residual_ST1 = SpatialTransformer(1024, 8, 128, depth=1, context_dim=2048, 
                               disable_self_attn=False, use_linear=True, 
                               use_checkpoint=False)
        self.global_residual_ST2 = SpatialTransformer(1024, 8, 128, depth=1, context_dim=2048, 
                               disable_self_attn=False, use_linear=True, 
                               use_checkpoint=False)
        self.global_residual_conv = zero_module(conv_nd(2, 1024, 1024, 1, padding=0))
        
        # project_out
        self.linear = EqualLinear(1024, 768, lr_mul=0.01, activation="fused_lrelu")

        if self.warp_feature:
            # upsample for 3D convolution
            # upsample_channels: [3072, 2048, 1024, 768]
            self.upsample_2d =  nn.Sequential()
            for i in range(len(upsample_channels) - 1):
                self.upsample_2d.add_module('upsample' + str(i), UpBlock2d(upsample_channels[i], upsample_channels[i+1]))

            # resblocks3D for 3D convolution
            self.resblocks_3d = nn.Sequential()
            for i in range(num_resblocks):
                self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

            # downsample for output
            # downsample_channels: [512, 512, 512, 768]
            self.downsample_2d =  nn.Sequential()
            for i in range(len(downsample_channels) - 1):
                self.downsample_2d.add_module('downsample' + str(i), DownBlock2d(downsample_channels[i], downsample_channels[i+1]))


    def forward(self, data_for_global, data_for_id, mesh_source = None, mesh_target = None):
        # data_for_id: tensor of shape:(B, 1024)
        # data_for_global: tensor of shape:(B, 3, 224, 224), 0~1,
        # mesh_source, mesh_target : tensor of shape:(B, 3, 5023)
        
        # get 3d feature
        id_feature = data_for_id.unsqueeze(-1).unsqueeze(-1) # (B, 1024, 1, 1)

        global_feature = self.global_extractor(data_for_global).unsqueeze(1) # global_feature: (B, 1, 2048)

        # todo: abandon this
        if self.warp_feature:
            combined_information = combined_information.view(combined_information.shape[0], -1, 1, 1) # combined_information: (B, 3072, 1, 1)

            upsample_feature = self.upsample_2d(combined_information) # upsample_feature: (B, 512, 8, 8)
            
            bs, c, h, w = upsample_feature.shape
            feature_3d = upsample_feature.view(bs, self.reshape_channel, self.reshape_depth, h ,w) # feature_3d: (B, 32 ,16, 8, 8)
            feature_3d = self.resblocks_3d(feature_3d)

            # get flow from two meshs
            flow_field, k_num = create_sparse_motions(feature_3d, mesh_target, mesh_source) # flow_field: (B, k_num + 1, 32, 16, 8, 8, 3)
            flow_field = flow_field.sum(dim=1) / (k_num + 1)           # flow_field: (B, 32, 16, 8, 8, 3)

            # warp 3d feature
            deformed_feature_3d = F.grid_sample(feature_3d, flow_field, padding_mode='border')   # (Batchsize, C, D, H, W)

            # return 2d feature
            bs, c, d, h, w = deformed_feature_3d.shape
            feature_2d = deformed_feature_3d.view((bs, c*d, h, w)) # (Batchsize, 512, 8, 8)
            feature_2d = self.downsample_2d(feature_2d) # (Batchsize, 768, 1, 1)
            feature_2d = feature_2d.view(bs, -1)

            return feature_2d
        
        else:
            global_residual = self.global_residual_ST1(id_feature, global_feature) # global_residual: (B, 1024, 1, 1)
            global_residual = self.global_residual_ST2(global_residual, global_feature) # global_residual: (B, 1024, 1, 1)
            global_residual = self.global_residual_conv(global_residual) # global_residual: (B, 1024, 1, 1)

            feature = id_feature + global_residual # feature: (B, 1024, 1, 1)
            feature = feature.transpose(1, 2).squeeze(-1) # feature: (B, 1, 1024)
            feature = self.mapper(feature) # feature: (B, 1, 1024)
            feature = self.linear(feature.squeeze(1)).unsqueeze(1) # feature: (B, 1, 768)

            return feature
    
