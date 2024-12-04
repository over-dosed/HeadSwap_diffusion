"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""


import os, sys
import torch
import torchvision
import torch.nn.functional as F
import adabound
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
import numpy as np
# from time import time
from skimage.io import imread
import cv2
from pathlib import Path

from HSD.utils.emoca.models.Renderer import SRenderY
from HSD.utils.emoca.models.DecaEncoder import ResnetEncoder, SecondHeadResnet, SwinEncoder
from HSD.utils.emoca.models.DecaDecoder import Generator, GeneratorAdaIn
from HSD.utils.emoca.models.DecaFLAME import FLAME, FLAMETex
from HSD.utils.emoca.models.EmotionMLP import EmotionMLP

import HSD.utils.emoca.layers.losses.DecaLosses as lossfunc
import HSD.utils.emoca.utils.DecaUtils as util

torch.backends.cudnn.benchmark = True
from enum import Enum
from HSD.utils.emoca.utils.other import class_from_str
from HSD.utils.emoca.layers.losses.VGGLoss import VGG19Loss
from omegaconf import OmegaConf, open_dict

import pytorch_lightning.plugins.environments.lightning_environment as le


class DecaMode(Enum):
    COARSE = 1 # when switched on, only coarse part of DECA-based networks is used
    DETAIL = 2 # when switched on, only coarse and detail part of DECA-based networks is used 


class DecaModule(LightningModule):
    """
    DecaModule is a PL module that implements DECA-inspired face reconstruction networks. 
    """

    def __init__(self, model_params, learning_params, inout_params, stage_name = ""):
        """
        :param model_params: a DictConfig of parameters about the model itself
        :param learning_params: a DictConfig of parameters corresponding to the learning process (such as optimizer, lr and others)
        :param inout_params: a DictConfig of parameters about input and output (where checkpoints and visualizations are saved)
        """
        super().__init__()
        self.learning_params = learning_params
        self.inout_params = inout_params

        # detail conditioning - what is given as the conditioning input to the detail generator in detail stage training
        if 'detail_conditioning' not in model_params.keys():
            # jaw, expression and detail code by default
            self.detail_conditioning = ['jawpose', 'expression', 'detail'] 
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detail_conditioning = self.detail_conditioning
        else:
            self.detail_conditioning = model_params.detail_conditioning

        # deprecated and is not used
        if 'detailemo_conditioning' not in model_params.keys():
            self.detailemo_conditioning = []
            OmegaConf.set_struct(model_params, True)
            with open_dict(model_params):
                model_params.detailemo_conditioning = self.detailemo_conditioning
        else:
            self.detailemo_conditioning = model_params.detailemo_conditioning

        supported_conditioning_keys = ['identity', 'jawpose', 'expression', 'detail', 'detailemo']
        
        for c in self.detail_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")
        for c in self.detailemo_conditioning:
            if c not in supported_conditioning_keys:
                raise ValueError(f"Conditioning on '{c}' is not supported. Supported conditionings: {supported_conditioning_keys}")

        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)

        self.mode = DecaMode[str(model_params.mode).upper()]
        self.stage_name = stage_name
        if self.stage_name is None:
            self.stage_name = ""
        if len(self.stage_name) > 0:
            self.stage_name += "_"

    def _encode_flame(self, images):
        if self.mode == DecaMode.COARSE or \
                (self.mode == DecaMode.DETAIL and self.deca.config.train_coarse):
            # forward pass with gradients (for coarse stage (used), or detail stage with coarse training (not used))
            parameters = self.deca._encode_flame(images)
        elif self.mode == DecaMode.DETAIL:
            # in detail stage, the coarse forward pass does not need gradients
            with torch.no_grad():
                parameters = self.deca._encode_flame(images)
        else:
            raise ValueError(f"Invalid EMOCA Mode {self.mode}")
        code_list = self.deca.decompose_code(parameters)
        shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        return shapecode, texcode, expcode, posecode, cam, lightcode

    def encode(self, batch):
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, ring_size, 3, image_size, image_size]. 
        For a training forward pass, additional corresponding data are necessery such as 'landmarks' and 'masks'. 
        For a testing pass, the images suffice. 
        :param training: Whether the forward pass is for training or testing.
        """

        # forward pass of the coarse encoder
        shapecode, texcode, expcode, posecode, cam, lightcode = self._encode_flame(batch)

        return shapecode, texcode, expcode, posecode, cam, lightcode

    @property
    def process(self):
        if not hasattr(self,"process_"):
            import psutil
            self.process_ = psutil.Process(os.getpid())
        return self.process_


class DECA(torch.nn.Module):
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, config):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()
        
        # ID-MRF perceptual loss (kept here from the original DECA implementation)
        self.perceptual_loss = None
        
        # Face Recognition loss
        self.id_loss = None

        # VGG feature loss
        self.vgg_loss = None
        
        self._reconfigure(config)
        self._reinitialize()

    def _reconfigure(self, config):
        self.config = config
        
        self.n_param = config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        # identity-based detail code 
        self.n_detail = config.n_detail
        # emotion-based detail code (deprecated, not use by DECA or EMOCA)
        self.n_detail_emo = config.n_detail_emo if 'n_detail_emo' in config.keys() else 0

        # count the size of the conidition vector
        if 'detail_conditioning' in self.config.keys():
            self.n_cond = 0
            if 'globalpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'jawpose' in self.config.detail_conditioning:
                self.n_cond += 3
            if 'identity' in self.config.detail_conditioning:
                self.n_cond += config.n_shape
            if 'expression' in self.config.detail_conditioning:
                self.n_cond += config.n_exp
        else:
            self.n_cond = 3 + config.n_exp

        self.mode = DecaMode[str(config.mode).upper()]
        self._create_detail_generator()
        self._init_deep_losses()
        self._setup_neural_rendering()

    def _reinitialize(self):
        self._create_model()
        self._setup_renderer()
        self._init_deep_losses()
        self.face_attr_mask = util.load_local_mask(image_size=self.config.uv_size, mode='bbx')

    def _init_deep_losses(self):
        """
        Initialize networks for deep losses
        """
        # TODO: ideally these networks should be moved out the DECA class and into DecaModule, 
        # but that would break backwards compatility with the original DECA and would not be able to load DECA's weights
        if 'mrfwr' not in self.config.keys() or self.config.mrfwr == 0:
            self.perceptual_loss = None
        else:
            if self.perceptual_loss is None:
                self.perceptual_loss = lossfunc.IDMRFLoss().eval()
                self.perceptual_loss.requires_grad_(False)  # TODO, move this to the constructor

        if 'idw' not in self.config.keys() or self.config.idw == 0:
            self.id_loss = None
        else:
            if self.id_loss is None:
                id_metric = self.config.id_metric if 'id_metric' in self.config.keys() else None
                id_trainable = self.config.id_trainable if 'id_trainable' in self.config.keys() else False
                self.id_loss_start_step = self.config.id_loss_start_step if 'id_loss_start_step' in self.config.keys() else 0
                self.id_loss = lossfunc.VGGFace2Loss(self.config.pretrained_vgg_face_path, id_metric, id_trainable)
                self.id_loss.freeze_nontrainable_layers()

        if 'vggw' not in self.config.keys() or self.config.vggw == 0:
            self.vgg_loss = None
        else:
            if self.vgg_loss is None:
                vgg_loss_batch_norm = 'vgg_loss_batch_norm' in self.config.keys() and self.config.vgg_loss_batch_norm
                self.vgg_loss = VGG19Loss(dict(zip(self.config.vgg_loss_layers, self.config.lambda_vgg_layers)), batch_norm=vgg_loss_batch_norm).eval()
                self.vgg_loss.requires_grad_(False) # TODO, move this to the constructor

    def _setup_renderer(self):
        self.render = SRenderY(self.config.image_size, obj_filename=self.config.topology_path,
                               uv_size=self.config.uv_size)  # .to(self.device)
        # face mask for rendering details
        mask = imread(self.config.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        mask = imread(self.config.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [self.config.uv_size, self.config.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # displacement mask is deprecated and not used by DECA or EMOCA
        if 'displacement_mask' in self.config.keys():
            displacement_mask_ = 1-np.load(self.config.displacement_mask).astype(np.float32)
            # displacement_mask_ = np.load(self.config.displacement_mask).astype(np.float32)
            displacement_mask_ = torch.from_numpy(displacement_mask_)[None, None, ...].contiguous()
            displacement_mask_ = F.interpolate(displacement_mask_, [self.config.uv_size, self.config.uv_size])
            self.register_buffer('displacement_mask', displacement_mask_)

        ## displacement correct
        if os.path.isfile(self.config.fixed_displacement_path):
            fixed_dis = np.load(self.config.fixed_displacement_path)
            fixed_uv_dis = torch.tensor(fixed_dis).float()
        else:
            fixed_uv_dis = torch.zeros([512, 512]).float()
        self.register_buffer('fixed_uv_dis', fixed_uv_dis)

    def uses_texture(self): 
        # if 'use_texture' in self.config.keys():
        #     return self.config.use_texture
        return True # true by default

    def _disable_texture(self, remove_from_model=False): 
        self.config.use_texture = False
        if remove_from_model:
            self.flametex = None

    def _enable_texture(self):
        self.config.use_texture = True

    def _has_neural_rendering(self):
        return hasattr(self.config, "neural_renderer") and bool(self.config.neural_renderer)

    def _setup_neural_rendering(self):
        if self._has_neural_rendering():
            if self.config.neural_renderer.class_ == "StarGAN":
                from .StarGAN import StarGANWrapper
                print("Creating StarGAN neural renderer")
                self.image_translator = StarGANWrapper(self.config.neural_renderer.cfg, self.config.neural_renderer.stargan_repo)
            else:
                raise ValueError(f"Unsupported neural renderer class '{self.config.neural_renderer.class_}'")

            if self.image_translator.background_mode == "input":
                if self.config.background_from_input not in [True, "input"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "Background should be inpainted from the input")
            elif self.image_translator.background_mode == "black":
                if self.config.background_from_input not in [False, "black"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "Background should be black.")
            elif self.image_translator.background_mode == "none":
                if self.config.background_from_input not in ["none"]:
                    raise NotImplementedError("The background mode of the neural renderer and deca is not synchronized. "
                                              "The background should not be handled")
            else:
                raise NotImplementedError(f"Unsupported mode of the neural renderer backroungd: "
                                          f"'{self.image_translator.background_mode}'")

    def _create_detail_generator(self):
        #backwards compatibility hack:
        if hasattr(self, 'D_detail'):
            if (not "detail_conditioning_type" in self.config.keys() or  self.config.detail_conditioning_type == "concat") \
                and isinstance(self.D_detail, Generator):
                return
            if self.config.detail_conditioning_type == "adain" and isinstance(self.D_detail, GeneratorAdaIn):
                return
            print("[WARNING]: We are reinitializing the detail generator!")
            del self.D_detail # just to make sure we free the CUDA memory, probably not necessary

        if not "detail_conditioning_type" in self.config.keys() or str(self.config.detail_conditioning_type).lower() == "concat":
            # concatenates detail latent and conditioning (this one is used by DECA/EMOCA)
            print("Creating classic detail generator.")
            self.D_detail = Generator(latent_dim=self.n_detail + self.n_detail_emo + self.n_cond, out_channels=1, out_scale=0.01,
                                      sample_mode='bilinear')
        elif str(self.config.detail_conditioning_type).lower() == "adain":
            # conditioning passed in through adain layers (this one is experimental and not currently used)
            print("Creating AdaIn detail generator.")
            self.D_detail = GeneratorAdaIn(self.n_detail + self.n_detail_emo,  self.n_cond, out_channels=1, out_scale=0.01,
                                      sample_mode='bilinear')
        else:
            raise NotImplementedError(f"Detail conditioning invalid: '{self.config.detail_conditioning_type}'")

    def _create_model(self):
        # 1) build coarse encoder
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in self.config.keys():
            e_flame_type = self.config.e_flame_type

        if e_flame_type == 'ResnetEncoder':
            self.E_flame = ResnetEncoder(outsize=self.n_param)
        elif e_flame_type[:4] == 'swin':
            self.E_flame = SwinEncoder(outsize=self.n_param, img_size=self.config.image_size, swin_type=e_flame_type)
        else:
            raise ValueError(f"Invalid 'e_flame_type' = {e_flame_type}")

        self.flame = FLAME(self.config)

        if self.uses_texture():
            self.flametex = FLAMETex(self.config)
        else: 
            self.flametex = None

        # 2) build detail encoder
        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in self.config.keys():
            e_detail_type = self.config.e_detail_type

        if e_detail_type == 'ResnetEncoder':
            self.E_detail = ResnetEncoder(outsize=self.n_detail + self.n_detail_emo)
        elif e_flame_type[:4] == 'swin':
            self.E_detail = SwinEncoder(outsize=self.n_detail + self.n_detail_emo, img_size=self.config.image_size, swin_type=e_detail_type)
        else:
            raise ValueError(f"Invalid 'e_detail_type'={e_detail_type}")
        self._create_detail_generator()
        # self._load_old_checkpoint()

    def _get_coarse_trainable_parameters(self):
        print("Add E_flame.parameters() to the optimizer")
        return list(self.E_flame.parameters())

    def _get_detail_trainable_parameters(self):
        trainable_params = []
        if self.config.train_coarse:
            trainable_params += self._get_coarse_trainable_parameters()
            print("Add E_flame.parameters() to the optimizer")
        trainable_params += list(self.E_detail.parameters())
        print("Add E_detail.parameters() to the optimizer")
        trainable_params += list(self.D_detail.parameters())
        print("Add D_detail.parameters() to the optimizer")
        return trainable_params

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_flame.train()
                # print("Setting E_flame to train")
                self.E_detail.eval()
                # print("Setting E_detail to eval")
                self.D_detail.eval()
                # print("Setting D_detail to eval")
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    # print("Setting E_flame to train")
                    self.E_flame.train()
                else:
                    # print("Setting E_flame to eval")
                    self.E_flame.eval()
                self.E_detail.train()
                # print("Setting E_detail to train")
                self.D_detail.train()
                # print("Setting D_detail to train")
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_flame.eval()
            # print("Setting E_flame to eval")
            self.E_detail.eval()
            # print("Setting E_detail to eval")
            self.D_detail.eval()
            # print("Setting D_detail to eval")

        # these are set to eval no matter what, they're never being trained (the FLAME shape and texture spaces are pretrained)
        self.flame.eval()
        if self.flametex is not None:
            self.flametex.eval()
        return self


    def _load_old_checkpoint(self):
        """
        Loads the DECA model weights from the original DECA implementation: 
        https://github.com/YadiraF/DECA 
        """
        if self.config.resume_training:
            model_path = self.config.pretrained_modelpath
            print(f"Loading model state from '{model_path}'")
            checkpoint = torch.load(model_path)
            # model
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            # util.copy_state_dict(self.opt.state_dict(), checkpoint['opt']) # deprecate
            # detail model
            if 'E_detail' in checkpoint.keys():
                util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
                util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            # training state
            self.start_epoch = 0  # checkpoint['epoch']
            self.start_iter = 0  # checkpoint['iter']
        else:
            print('Start training from scratch')
            self.start_epoch = 0
            self.start_iter = 0

    def _encode_flame(self, images):
        return self.E_flame(images)

    def decompose_code(self, code):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        code_list = []
        num_list = [self.config.n_shape, self.config.n_tex, self.config.n_exp, self.config.n_pose, self.config.n_cam,
                    self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals, detach=True):
        """
        Converts the displacement uv map (uv_z) and coarse_verts to a normal map coarse_normals. 
        """
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)#.detach()
        if detach:
            uv_coarse_vertices = uv_coarse_vertices.detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals)#.detach()
        if detach:
            uv_coarse_normals = uv_coarse_normals.detach()

        uv_z = uv_z * self.uv_face_eye_mask

        # detail vertices = coarse vertice + predicted displacement*normals + fixed displacement*normals
        uv_detail_vertices = uv_coarse_vertices + \
                             uv_z * uv_coarse_normals + \
                             self.fixed_uv_dis[None, None, :,:] * uv_coarse_normals #.detach()

        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        # uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        # uv_detail_normals = util.gaussian_blur(uv_detail_normals)
        return uv_detail_normals, uv_coarse_vertices

    def visualize(self, visdict, savepath, catdim=1):
        grids = {}
        for key in visdict:
            # print(key)
            if visdict[key] is None:
                continue
            grids[key] = torchvision.utils.make_grid(
                F.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), catdim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath is not None:
            cv2.imwrite(savepath, grid_image)
        return grid_image

    def create_mesh(self, opdict, dense_template):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        if 'uv_texture_gt' in opdict.keys():
            texture = util.tensor2image(opdict['uv_texture_gt'][i])
        else:
            texture = None
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        if 'uv_detail_normals' in opdict.keys():
            normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
            # upsample mesh, save detailed mesh
            texture = texture[:, :, [2, 1, 0]]
            normals = opdict['normals'][i].cpu().numpy()
            displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
            dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces,
                                                                           displacement_map, texture, dense_template)
        else:
            normal_map = None
            dense_vertices = None
            dense_colors  = None
            dense_faces  = None

        return vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors


    def save_obj(self, filename, opdict, dense_template, mode ='detail'):
        if mode not in ['coarse', 'detail', 'both']:
            raise ValueError(f"Invalid mode '{mode}. Expected modes are: 'coarse', 'detail', 'both'")

        vertices, faces, texture, uvcoords, uvfaces, normal_map, dense_vertices, dense_faces, dense_colors \
            = self.create_mesh(opdict, dense_template)

        if mode == 'both':
            if isinstance(filename, list):
                filename_coarse = filename[0]
                filename_detail = filename[1]
            else:
                filename_coarse = filename
                filename_detail = filename.replace('.obj', '_detail.obj')
        elif mode == 'coarse':
            filename_coarse = filename
        else:
            filename_detail = filename

        if mode in ['coarse', 'both']:
            util.write_obj(str(filename_coarse), vertices, faces,
                            texture=texture,
                            uvcoords=uvcoords,
                            uvfaces=uvfaces,
                            normal_map=normal_map)

        if mode in ['detail', 'both']:
            util.write_obj(str(filename_detail),
                            dense_vertices,
                            dense_faces,
                            colors = dense_colors,
                            inverse_face_order=True)


from HSD.utils.emoca.models.EmoNetRegressor import EmoNetRegressor, EmonetRegressorStatic


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality. 
    Such as a separate expression decoder and related.
    """

    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        # E_flame should be fixed for expression EMOCA
        self.E_flame.requires_grad_(False)
        
        # 2) add expression decoder
        if self.config.expression_backbone == 'deca_parallel':
            ## a) Attach a parallel flow of FCs onto the original DECA coarse backbone. (Only the second FC head is trainable)
            self.E_expression = SecondHeadResnet(self.E_flame, self.n_exp_param, 'same')
        elif self.config.expression_backbone == 'deca_clone':
            ## b) Clones the original DECA coarse decoder (and the entire decoder will be trainable) - This is in final EMOCA.
            #TODO this will only work for Resnet. Make this work for the other backbones (Swin) as well.
            self.E_expression = ResnetEncoder(self.n_exp_param)
            # clone parameters of the ResNet
            self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        elif self.config.expression_backbone == 'emonet_trainable':
            # Trainable EmoNet instead of Resnet (deprecated)
            self.E_expression = EmoNetRegressor(self.n_exp_param)
        elif self.config.expression_backbone == 'emonet_static':
            # Frozen EmoNet with a trainable head instead of Resnet (deprecated)
            self.E_expression = EmonetRegressorStatic(self.n_exp_param)
        else:
            raise ValueError(f"Invalid expression backbone: '{self.config.expression_backbone}'")

    def _get_coarse_trainable_parameters(self):
        print("Add E_expression.parameters() to the optimizer")
        return list(self.E_expression.parameters())

    def _reconfigure(self, config):
        super()._reconfigure(config)
        self.n_exp_param = self.config.n_exp

        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            self.n_exp_param += self.config.n_pose
        elif self.config.exp_deca_global_pose or self.config.exp_deca_jaw_pose:
            self.n_exp_param += 3

    def _encode_flame(self, images):
        if self.config.expression_backbone == 'deca_parallel':
            #SecondHeadResnet does the forward pass for shape and expression at the same time
            return self.E_expression(images)
        # other regressors have to do a separate pass over the image
        deca_code = super()._encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]

        deca_code_list = super().decompose_code(deca_code)
        # shapecode, texcode, expcode, posecode, cam, lightcode = deca_code_list
        exp_idx = 2
        pose_idx = 3

        #TODO: clean this if-else block up
        if self.config.exp_deca_global_pose and self.config.exp_deca_jaw_pose:
            exp_code = expdeca_code[:, :self.config.n_exp]
            pose_code = expdeca_code[:, self.config.n_exp:]
            deca_code_list[exp_idx] = exp_code
            deca_code_list[pose_idx] = pose_code
        elif self.config.exp_deca_global_pose:
            # global pose from ExpDeca, jaw pose from EMOCA
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_exp_deca, pose_code_deca[:,3:]], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        elif self.config.exp_deca_jaw_pose:
            # global pose from EMOCA, jaw pose from ExpDeca
            pose_code_exp_deca = expdeca_code[:, self.config.n_exp:]
            pose_code_deca = deca_code_list[pose_idx]
            deca_code_list[pose_idx] = torch.cat([pose_code_deca[:, :3], pose_code_exp_deca], dim=1)
            exp_code = expdeca_code[:, :self.config.n_exp]
            deca_code_list[exp_idx] = exp_code
        else:
            exp_code = expdeca_code
            deca_code_list[exp_idx] = exp_code

        return deca_code_list

    def train(self, mode: bool = True):
        super().train(mode)

        # for expression deca, we are not training the resnet feature extractor plus the identity/light/texture regressor
        self.E_flame.eval()

        if mode:
            if self.mode == DecaMode.COARSE:
                self.E_expression.train()
                # print("Setting E_expression to train")
                self.E_detail.eval()
                # print("Setting E_detail to eval")
                self.D_detail.eval()
                # print("Setting D_detail to eval")
            elif self.mode == DecaMode.DETAIL:
                if self.config.train_coarse:
                    # print("Setting E_flame to train")
                    self.E_expression.train()
                else:
                    # print("Setting E_flame to eval")
                    self.E_expression.eval()
                self.E_detail.train()
                # print("Setting E_detail to train")
                self.D_detail.train()
            else:
                raise ValueError(f"Invalid mode '{self.mode}'")
        else:
            self.E_expression.eval()
            self.E_detail.eval()
            self.D_detail.eval()
        return self


def instantiate_deca(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
    """
    Function that instantiates a DecaModule from checkpoint or config
    """

    if checkpoint is None:
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, prefix)
        if cfg.model.resume_training:
            # This load the DECA model weights from the original DECA release
            print("[WARNING] Loading EMOCA checkpoint pretrained by the old code")
            deca.deca._load_old_checkpoint()
    else:
        checkpoint_kwargs = checkpoint_kwargs or {}
        deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
        if stage == 'train':
            mode = True
        else:
            mode = False
        deca.reconfigure(cfg.model, cfg.inout, cfg.learning, prefix, downgrade_ok=True, train=mode)
    return deca
