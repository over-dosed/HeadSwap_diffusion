import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel
from einops import rearrange

from modules.util import EqualLinear
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    zero_module,
)
from ldm.modules.encoders.xf import Transformer

class id_residual(nn.Module):

    def __init__(self, spatial_depth = 2):
        super(id_residual, self).__init__()

        self.spatial_depth = spatial_depth
        self.id_residual_ST = SpatialTransformer(1024, 8, 128, depth=spatial_depth, 
                                                        context_dim=[1024]*spatial_depth, disable_self_attn=False, 
                                                        use_linear=True, use_checkpoint=False)

    def forward(self, CLIP_output, id_feature):
        # CLIP_output : (B, 257, 1024)
        # id_feature : (B, 1, 1024)

        input_id = id_feature
        input = rearrange(CLIP_output, 'b t c -> b c t 1').contiguous() # (B, 1024, 257, 1)
        
        id_residual = self.id_residual_ST(input, input_id) # (B, 1024, 257, 1)
        id_residual = rearrange(id_residual, 'b c t 1 -> b t c').contiguous() # (B, 257, 1024)

        return id_residual

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class Feature_Branch(nn.Module):

    def __init__(self, CLIP_version="openai/clip-vit-large-patch14"):
        super().__init__() 
        self.transformer = CLIPVisionModel.from_pretrained(CLIP_version)

        # project_in
        self.id_proj_in = EqualLinear(512, 1024, lr_mul=0.01, activation="fused_lrelu")

        # mapper
        self.id_mapper = Transformer(1, 1024, 5, 8)
        self.global_mapper = Transformer(257, 1024, 5, 8)

        # id_residual
        self.id_residual_block = id_residual(spatial_depth = 2)

        # normalize
        self.id_norm = nn.LayerNorm(1024)
        self.global_norm = nn.LayerNorm(1024)

        # project_out
        self.id_proj_out = nn.Linear(1024, 768)
        self.global_proj_out = nn.Linear(1024, 768)

        self.device = 'cuda:0'
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.id_proj_in.parameters():
            param.requires_grad = True
        for param in self.id_mapper.parameters():
            param.requires_grad = True
        for param in self.global_mapper.parameters():
            param.requires_grad = True
        for param in self.id_residual_block.parameters():
            param.requires_grad = True
        for param in self.id_norm.parameters():
            param.requires_grad = True
        for param in self.global_norm.parameters():
            param.requires_grad = True
        for param in self.global_proj_out.parameters():
            param.requires_grad = True
        for param in self.id_proj_out.parameters():
            param.requires_grad = True

    def forward(self, image, id_feature):
        # id_feature: tensor of shape:(B, 512)
        # image: tensor of shape:(B, 3, 224, 224), 0~1,
        
        with torch.no_grad():
            outputs = self.transformer(pixel_values=image)
            z = outputs.last_hidden_state # z : (B, 257, 1024)

        # id proj_in
        id_feature = self.id_proj_in(id_feature) # id_feature: (B, 1024)
        id_feature = id_feature.unsqueeze(1) # (B, 1, 1024)

        # map
        id_feature = self.id_mapper(id_feature) # id_feature: (B, 1, 1024)
        z = self.global_mapper(z) # z : (B, 257, 1024)

        # calculate id residual
        id_residual = self.id_residual_block(z, id_feature) # id_residual: (B, 257, 1024)
        z =  z + id_residual # z: (B, 257, 1024)

        # norm & proj_out
        id_feature = self.id_norm(id_feature)
        id_feature = self.id_proj_out(id_feature)
        z = self.global_norm(z)
        z = self.global_proj_out(z)

        outdict = {'c_crossattn': z, 'c_adapter': id_feature}
        return outdict