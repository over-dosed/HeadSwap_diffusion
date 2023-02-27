import os

import torch
import torch.nn as nn

from utils.ResNeXt.resnext_50_32x4d import ResNext
from utils.arcface_pytorch.models.resnet import resnet_face18


class Feature_Branch(nn.Module):




    def __init__(self, load_first_train, arcface_path = None, resNext_path = None, fix_id_extractor = True, fix_global_extractor = False):
        super().__init__() 
        # load_first_train is to decide whether load weight of id&global extractor or not

        # argface for id extractor
        self.id_extractor = resnet_face18(False)
        if fix_id_extractor:
            self.id_extractor.eval()
        else:
            self.id_extractor.train()

        # ResNeXt for global extractor
        self.global_extractor = ResNext().network()
        if fix_global_extractor:
            self.global_extractor.eval()
        else:
            self.global_extractor.train()

        # resblocks3D for 3D convolution
        

        if load_first_train:
            load_first_train(arcface_path, resNext_path)
        

    def forward(self, x):
        x = self.linear(x) # 线性变换
        x = self.relu(x) # 激活函数
        return x
    
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