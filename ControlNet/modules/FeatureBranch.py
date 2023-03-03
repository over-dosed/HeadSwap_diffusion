import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ResNeXt.resnext_50_32x4d import ResNext
from utils.arcface_pytorch.models.resnet import resnet_face18
from modules.util import ResBlock3d, UpBlock2d, DownBlock2d, create_sparse_motions, EqualLinear

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Feature_Branch(nn.Module):

    def __init__(self, 
                 reshape_channel, 
                 reshape_depth, 
                 num_resblocks, 
                 upsample_channels, 
                 downsample_channels, 
                 arcface_path = None, 
                 resNext_path = None, 
                 fix_id_extractor = True,
                 warp_feature = False,
                 linear_channels = None):
        super().__init__() 
        self.fix_id_extractor = fix_id_extractor
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth
        self.warp_feature = warp_feature

        # argface for id extractor
        self.id_extractor = resnet_face18(False)
        if fix_id_extractor:
            self.id_extractor.eval()
            self.id_extractor.train = disabled_train
            for param in self.id_extractor.parameters():
                param.requires_grad = False
        else:
            self.id_extractor.train()

        # ResNeXt for global extractor
        self.global_extractor = ResNext().network()

        # for no warp , linear only
        # linear_channels: [3072, 2048, 1024, 768]
        self.linear = nn.Sequential()
        for i in range(len(linear_channels) - 1):
            self.linear.add_module('linear' + str(i), EqualLinear(linear_channels[i], linear_channels[i+1], lr_mul=0.01, activation="fused_lrelu"))

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

        # load weight for id&gloabl models
        if arcface_path is not None or resNext_path is not None:
            self.load_first_train(arcface_path, resNext_path)
        

    def forward(self, data_for_id, data_for_global, mesh_source, mesh_target):
        # data_for_id: tensor of shape:(B * 2, 1, 128, 128), -1~1, float32, check for more information: DataPreprocess/getIdInformation.py/process_a_image
        # data_for_global: tensor of shape:(B, 3, 224, 224), 0~1,
        # mesh_source, mesh_target : tensor of shape:(B, 3, 5023)
        
        # get 3d feature
        if data_for_id.dim() == 2:         # if already extracted id_feature
            id_feature = data_for_id
        else: id_feature = self.id_extractor(data_for_id)  # id_feature: (B, 1024)
        global_feature = self.global_extractor(data_for_global) # global_feature: (B, 2048)
        combined_information = torch.concat((id_feature, global_feature), dim=1) # combined_information: (B, 3072)

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
            feature = self.linear(combined_information)
            return feature
            
    
    def load_first_train(self, arcface_path = None, resNext_path = None):
        # for the first train, you mind wish to load the initial weight of id&global extractor
        # this method only load state_dict of the two models
        
        if arcface_path is not None:
            arcface_state_dict = torch.load(arcface_path)
            self.id_extractor.load_state_dict(state_dict= arcface_state_dict)
        if resNext_path is not None:
            resNext_state_dict = torch.load(resNext_path)
            self.global_extractor.load_state_dict(state_dict= resNext_state_dict)
        
        return
    
