import cv2
from PIL import Image

import os
import os.path as osp

import numpy as np
import pickle
import random

import torch
from torch.utils.data import Dataset
import torchvision

from modules.ConditionBranch import Condition_Branch

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

class HSD_Dataset(Dataset):
    def __init__(self, root_path, batch_size):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.bs = batch_size
        self.condition_branch = Condition_Branch()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_path = self.data[idx]
        condition_pkl_path = os.path.join(clip_path, '3DMM_condition.pkl')
        id_pkl_path = os.path.join(clip_path, 'id.pkl')

        with open(condition_pkl_path, 'rb') as f_condition:
            data_3dmm = pickle.load(f_condition)
        with open(id_pkl_path, 'rb') as f_id:
            id_feature = pickle.load(f_id)

        codedict, index = self.get_code_dict(data_3dmm, self.bs)

        source_image_list = []
        target_image_list = []
        mask_image_list = []
        bg_image_list = []
        id_feature_selected = np.zeros((self.bs, id_feature.shape[1])).astype(id_feature.dtype)

        # get images
        for i in range(len(index)):
            source_image_path = osp.join(clip_path, '{}.png'.format(str(index[i][0]).zfill(8)))
            target_image_path = osp.join(clip_path, '{}.png'.format(str(index[i][1]).zfill(8)))
            mask_image_path = osp.join(clip_path, 'mask_{}.jpg'.format(str(index[i][1]).zfill(8)))
            source_image_list.append(np.asarray(Image.open(source_image_path).convert("RGB").resize((224, 224))))
            target_image_list.append(np.asarray(Image.open(target_image_path).convert("RGB")))
            mask_image_list.append(np.asarray(Image.open(mask_image_path)))
            id_feature_selected[i] = id_feature[index[0]]

        # get masked images (background)
        for i in range(len(index)):
            mask = mask_image_list[i]
            # GaussianBlur again to reduce mask edge serrate
            mask = cv2.GaussianBlur(mask, (11, 11), 11)
            mask = np.where( (mask <= 0), 0, 255).astype('uint8')
            bg_image_list.append(cv2.bitwise_and(target_image_list[i], target_image_list[i], mask = 255 - mask))

        source_images = (np.asarray(source_image_list).astype(np.float32) / 255.0).transpose(0, 3, 1, 2)        # Normalize source images to [0, 1].
        target_images = (np.asarray(target_image_list).astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)  # Normalize target images to [-1, 1].
        mask_images = np.expand_dims(np.asarray(mask_image_list).astype(np.float32) / 255.0, axis=1)
        bg_images = np.asarray(bg_image_list).astype(np.float32) / 255.0

        bg_images = torch.from_numpy(bg_images.transpose(0, 3, 1, 2))
        bg_images = bg_images.cuda()

        # rendered_images = self.condition_branch(codedict, bg_images).detach().cpu().numpy()
        rendered_images = self.condition_branch(codedict, bg_images) # GPU, (B, 3, h, w)

        return dict(target=target_images, mask=mask_images, background=bg_images, source_global=source_images, source_id=id_feature_selected, hint=rendered_images)

    
    def get_code_dict(self, code_dict, batch_size = 4, pose_threshold = 0.02, loop_max_times = 20):
        # this method get original a clip code_dict as input
        # return the indexs selected randomly and the corresponding combined code_dict

        tforms = code_dict['tforms']
        shape_code = code_dict['shape']
        tex_code = code_dict['tex']
        exp_code = code_dict['exp']
        pose_code = code_dict['pose']
        cam_code = code_dict['cam']
        light_code = code_dict['light']

        tforms_new = torch.zeros(batch_size, tforms.shape[1], tforms.shape[2])
        shape_code_new = torch.zeros(batch_size, shape_code.shape[1])
        tex_code_new = torch.zeros(batch_size, tex_code.shape[1])
        exp_code_new = torch.zeros(batch_size, exp_code.shape[1])
        pose_code_new = torch.zeros(batch_size, pose_code.shape[1])
        cam_code_new = torch.zeros(batch_size, cam_code.shape[1])
        light_code_new = torch.zeros(batch_size, light_code.shape[1], light_code.shape[2])

        total_num = pose_code.shape[0]
        count = 0
        index = []

        # use loop_max_times to avoid endless loop
        loop_max_times = batch_size * loop_max_times
        loop_count = 0


        while True:
            a = random.randint(0, total_num-1)       # a for source
            b = random.randint(0, total_num-1)       # b for target

            if a == b :
                continue

            if abs(torch.mean(pose_code[a] - pose_code[b])) >= pose_threshold or loop_count >= loop_max_times:

                # get combined code
                tforms_new[count, :] = tforms[b]
                shape_code_new[count, :] = shape_code[a]
                tex_code_new[count, :] = tex_code[a]
                exp_code_new[count, :] = exp_code[b]
                pose_code_new[count, :] = pose_code[b]
                cam_code_new[count, :] = cam_code[b]
                light_code_new[count, :] = light_code[b]

                # get index
                index.append((a, b))

                count +=1

                if count == batch_size:
                    new_code_dict = {
                        'tforms':tforms_new.cuda(),
                        'shape':shape_code_new.cuda(),
                        'tex':tex_code_new.cuda(),
                        'exp':exp_code_new.cuda(),
                        'pose':pose_code_new.cuda(),
                        'cam':cam_code_new.cuda(),
                        'light':light_code_new.cuda()
                    }
                    return new_code_dict, index
            else:
                loop_count += 1
                continue

class HSD_Dataset_normal(Dataset):
    # this class is the normal way to get a item of a batch
    # the way up is unnormal way : once get a batch when call getitem()
    def __init__(self, root_path):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.condition_branch = Condition_Branch()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('start a getitem')
        clip_path = self.data[idx]
        condition_pkl_path = os.path.join(clip_path, '3DMM_condition.pkl')
        id_pkl_path = os.path.join(clip_path, 'id.pkl')

        with open(condition_pkl_path, 'rb') as f_condition:
            data_3dmm = pickle.load(f_condition)
        with open(id_pkl_path, 'rb') as f_id:
            id_feature = pickle.load(f_id)

        codedict, index = self.get_code_dict(data_3dmm)

        # get images
        source_image_path = osp.join(clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(clip_path, '{}.png'.format(str(index[1]).zfill(8)))
        mask_image_path = osp.join(clip_path, 'mask_{}.jpg'.format(str(index[1]).zfill(8)))
        source_image = Image.open(source_image_path).convert("RGB").resize((224,224))
        source_tensor = get_tensor_clip()(source_image).to(torch.float16)
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        mask_image = np.asarray(Image.open(mask_image_path))
        id_feature_selected = id_feature[index[0]]

        # get masked images (background)
        # GaussianBlur again to reduce mask edge serrate
        mask_image = cv2.GaussianBlur(mask_image, (11, 11), 11)
        mask_image = np.where( (mask_image <= 0), 0, 255).astype('uint8')
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        mask_image = np.expand_dims(mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        # rendered_images = self.condition_branch(codedict, bg_images).detach().cpu().numpy()
        rendered_images = self.condition_branch(codedict, bg_image).squeeze(0) # (3, h, w)
        # print('end a getitem')

        return dict(target=target_image, mask=mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, hint=rendered_images)

    
    def get_code_dict(self, code_dict, pose_threshold = 0.02, loop_max_times = 20):
        # this method get original a clip code_dict as input
        # return the indexs selected randomly and the corresponding combined code_dict

        tforms = code_dict['tforms']
        shape_code = code_dict['shape']
        tex_code = code_dict['tex']
        exp_code = code_dict['exp']
        pose_code = code_dict['pose']
        cam_code = code_dict['cam']
        light_code = code_dict['light']

        total_num = pose_code.shape[0]
        index = []

        # use loop_max_times to avoid endless loop
        loop_count = 0

        while True:
            a = random.randint(0, total_num-1)       # a for source
            b = random.randint(0, total_num-1)       # b for target

            if a == b :
                continue

            if abs(torch.mean(pose_code[a] - pose_code[b])) >= pose_threshold or loop_count >= loop_max_times:

                # get combined code
                tforms_new = tforms[b]
                shape_code_new = shape_code[a]
                tex_code_new = tex_code[a]
                exp_code_new = exp_code[b]
                pose_code_new = pose_code[b]
                cam_code_new = cam_code[b]
                light_code_new = light_code[b]

                # get index
                index = (a, b)

                new_code_dict = {
                    'tforms':tforms_new,
                    'shape':shape_code_new,
                    'tex':tex_code_new,
                    'exp':exp_code_new,
                    'pose':pose_code_new,
                    'cam':cam_code_new,
                    'light':light_code_new
                }
                return new_code_dict, index

            else:
                loop_count += 1
                continue


class HSD_Dataset_single(Dataset):
    # this class is the normal way to get a item of a batch
    # the way up is unnormal way : once get a batch when call getitem()
    def __init__(self, root_path):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.condition_branch = Condition_Branch()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('start a getitem')
        clip_path = '/data0/wc_data/VFHQ/test/Clip+2W7Bk7EcRMg+P0+C1+F3663-3770'
        condition_pkl_path = os.path.join(clip_path, '3DMM_condition.pkl')
        id_pkl_path = os.path.join(clip_path, 'id.pkl')

        with open(condition_pkl_path, 'rb') as f_condition:
            data_3dmm = pickle.load(f_condition)
        with open(id_pkl_path, 'rb') as f_id:
            id_feature = pickle.load(f_id)

        codedict, index = self.get_code_dict(data_3dmm)

        # get images
        source_image_path = osp.join(clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(clip_path, '{}.png'.format(str(index[1]).zfill(8)))
        mask_image_path = osp.join(clip_path, 'mask_{}.jpg'.format(str(index[1]).zfill(8)))
        source_image = Image.open(source_image_path).convert("RGB").resize((224,224))
        source_tensor = get_tensor_clip()(source_image).to(torch.float16)
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        mask_image = np.asarray(Image.open(mask_image_path))
        id_feature_selected = id_feature[index[0]]

        # get masked images (background)
        # GaussianBlur again to reduce mask edge serrate
        mask_image = cv2.GaussianBlur(mask_image, (11, 11), 11)
        mask_image = np.where( (mask_image <= 0), 0, 255).astype('uint8')
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        mask_image = np.expand_dims(mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        # rendered_images = self.condition_branch(codedict, bg_images).detach().cpu().numpy()
        rendered_images = self.condition_branch(codedict, bg_image).squeeze(0) # (3, h, w)
        # print('end a getitem')

        return dict(target=target_image, mask=mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, hint=rendered_images)

    
    def get_code_dict(self, code_dict, pose_threshold = 0.02, loop_max_times = 2):
        # this method get original a clip code_dict as input
        # return the indexs selected randomly and the corresponding combined code_dict

        tforms = code_dict['tforms']
        shape_code = code_dict['shape']
        tex_code = code_dict['tex']
        exp_code = code_dict['exp']
        pose_code = code_dict['pose']
        cam_code = code_dict['cam']
        light_code = code_dict['light']

        total_num = pose_code.shape[0]
        index = []

        # use loop_max_times to avoid endless loop
        loop_count = 0

        while True:
            a = 13       # a for source
            b = random.randint(0, total_num-1)       # b for target

            if a == b :
                continue

            if abs(torch.mean(pose_code[a] - pose_code[b])) >= pose_threshold or loop_count >= loop_max_times:

                # get combined code
                tforms_new = tforms[b]
                shape_code_new = shape_code[a]
                tex_code_new = tex_code[a]
                exp_code_new = exp_code[b]
                pose_code_new = pose_code[b]
                cam_code_new = cam_code[b]
                light_code_new = light_code[b]

                # get index
                index = (a, b)

                new_code_dict = {
                    'tforms':tforms_new,
                    'shape':shape_code_new,
                    'tex':tex_code_new,
                    'exp':exp_code_new,
                    'pose':pose_code_new,
                    'cam':cam_code_new,
                    'light':light_code_new
                }
                return new_code_dict, index

            else:
                loop_count += 1
                continue