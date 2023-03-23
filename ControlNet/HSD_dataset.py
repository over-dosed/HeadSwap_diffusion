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
    return torchvision.transforms.Compose(transform_list)\

def smooth_expand_mask(mask_image, ksize=None, sigmaX= None, sigmaY= None):
    # need to be applied in data preprocess, and drop this
    # GaussianBlur again to reduce mask edge serrateimport random
    if ksize is None or sigmaX is None or sigmaY is None:
        random_int = random.sample(range(-10, 30), 4)
        ksize=(33 + random_int[0]*2, 33 + random_int[1]*2)
        sigmaX= 43 + random_int[2]*2
        sigmaY= 43 + random_int[3]*2
    mask_image = cv2.GaussianBlur(mask_image, ksize, sigmaX=sigmaX, sigmaY = sigmaY)
    mask_image = np.where( (mask_image <= 0), 0, 255).astype('uint8')
    return mask_image

def mask_find_bbox(mask):
    mask_col = np.sum(mask, axis= 0)
    mask_row = np.sum(mask, axis= 1)

    left = np.where(mask_col >= 255)[0][0]
    right = np.where(mask_col >= 255)[0][-1]
    up = np.where(mask_row >= 255)[0][0]
    down = np.where(mask_row >= 255)[0][-1]

    bbox = [left, up, right, down]
    return bbox

def get_align_image(bbox, img, reshape_size = 224):
    h, w, _ = img.shape
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    center_point = [int((x1 + x2) / 2), int((y1 + y2) / 2)] ## recalculate the center point
    expand_size = int((y2 - y1) * 0.5) # expand_size -- half of the total crop size
    crop_size = expand_size * 2

    new_x1 = center_point[0] - expand_size
    new_x2 = center_point[0] + expand_size
    new_y1 = center_point[1] - expand_size
    new_y2 = center_point[1] + expand_size

    (crop_left, origin_left) = (0, new_x1) if new_x1 >= 0 else (-new_x1, 0)
    (crop_right, origin_right) = (crop_size, new_x2) if new_x2 <= w else (w-new_x1, w)
    (crop_top, origin_top) = (0, new_y1) if new_y1 >= 0 else (-new_y1, 0)
    (crop_bottom, origin_bottom) = (crop_size, new_y2) if new_y2 <= h else (h-new_y1, h)

    aligned_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    aligned_img[crop_top:crop_bottom, crop_left:crop_right] = img[origin_top:origin_bottom, origin_left:origin_right]
    aligned_img = Image.fromarray(aligned_img)
    aligned_img = aligned_img.resize((reshape_size, reshape_size))
    aligned_img = np.asarray(aligned_img)
    return aligned_img

class HSD_Dataset(Dataset):
    # this class is the normal way to get a item of a batch
    def __init__(self, root_path):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
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

        codedict, index = self.get_code_dict(data_3dmm)

        # get images paths
        source_image_path = osp.join(clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(clip_path, '{}.png'.format(str(index[1]).zfill(8)))
        source_mask_path = osp.join(clip_path, 'mask_{}.jpg'.format(str(index[0]).zfill(8)))
        target_mask_path = osp.join(clip_path, 'mask_{}.jpg'.format(str(index[1]).zfill(8)))

        # read images
        source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        source_mask_image = np.asarray(Image.open(source_mask_path))
        target_mask_image = np.asarray(Image.open(target_mask_path))

        # smooth masks (will be droped)
        source_mask_image = smooth_expand_mask(source_mask_image, ksize=(11, 11), sigmaX=11, sigmaY=11)
        target_mask_image = smooth_expand_mask(target_mask_image)

        # process source image
        source_image = cv2.bitwise_and(source_image, source_image, mask = source_mask_image) # get masked
        bbox = mask_find_bbox(source_mask_image)
        source_image = get_align_image(bbox=bbox, img=source_image) # get align & resized source image, (224, 224, 3), numpy, 0~255
        source_tensor = get_tensor_clip()(source_image.copy()).to(torch.float16)

        id_feature_selected = id_feature[index[0]]

        # get masked images (background)
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - target_mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        target_mask_image = np.expand_dims(target_mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        # rendered_images = self.condition_branch(codedict, bg_images).detach().cpu().numpy()
        rendered_images = self.condition_branch(codedict).squeeze(0) # (3, h, w)
        # print('end a getitem')

        return dict(target=target_image, mask=target_mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, hint=rendered_images)

    
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

class HSD_Dataset_cross(Dataset):
    # this class is the normal way to get a item of a batch
    # every item is cross id

    def __init__(self, root_path, lenth = 5):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.condition_branch = Condition_Branch()
        self.lenth = lenth

    def __len__(self):
        return self.lenth

    def __getitem__(self, idx):

        total_lenth = len(self.data)
        assert(total_lenth >= 2)

        source_idx = random.randint(0, total_lenth-1)
        target_idx = random.randint(0, total_lenth-1)
        while target_idx == source_idx:
            target_idx = random.randint(0, total_lenth-1)
        
        source_clip_path = self.data[source_idx]
        target_clip_path = self.data[target_idx]

        source_condition_pkl_path = os.path.join(source_clip_path, '3DMM_condition.pkl') 
        target_condition_pkl_path = os.path.join(target_clip_path, '3DMM_condition.pkl')
        id_pkl_path = os.path.join(source_clip_path, 'id.pkl')

        with open(source_condition_pkl_path, 'rb') as f_condition_source:
            source_data_3dmm = pickle.load(f_condition_source)
        with open(target_condition_pkl_path, 'rb') as f_condition_target:
            target_data_3dmm = pickle.load(f_condition_target)
        with open(id_pkl_path, 'rb') as f_id:
            id_feature = pickle.load(f_id)

        codedict, index = self.get_code_dict(source_data_3dmm, target_data_3dmm)

        # get images paths
        source_image_path = osp.join(source_clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(target_clip_path, '{}.png'.format(str(index[1]).zfill(8)))
        source_mask_path = osp.join(source_clip_path, 'mask_{}.jpg'.format(str(index[0]).zfill(8)))
        target_mask_path = osp.join(target_clip_path, 'mask_{}.jpg'.format(str(index[1]).zfill(8)))

        # read images
        source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        source_mask_image = np.asarray(Image.open(source_mask_path))
        target_mask_image = np.asarray(Image.open(target_mask_path))

        # smooth masks (will be droped)
        source_mask_image = smooth_expand_mask(source_mask_image, ksize=(11, 11), sigmaX=11, sigmaY=11)
        target_mask_image = smooth_expand_mask(target_mask_image, ksize=(55, 55), sigmaX=33, sigmaY=33)

        # process source image
        source_image = cv2.bitwise_and(source_image, source_image, mask = source_mask_image) # get masked
        bbox = mask_find_bbox(source_mask_image)
        source_image = get_align_image(bbox=bbox, img=source_image) # get align & resized source image, (224, 224, 3), numpy, 0~255
        source_tensor = get_tensor_clip()(source_image.copy()).to(torch.float16)

        id_feature_selected = id_feature[index[0]]

        # get masked images (background)
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - target_mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        target_mask_image = np.expand_dims(target_mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        # rendered_images = self.condition_branch(codedict, bg_images).detach().cpu().numpy()
        rendered_images = self.condition_branch(codedict).squeeze(0) # (3, h, w)
        # print('end a getitem')

        return dict(target=target_image, mask=target_mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, hint=rendered_images)

    
    def get_code_dict(self, source_data_3dmm, target_data_3dmm):
        # this method get original a clip code_dict as input
        # return the indexs selected randomly and the corresponding combined code_dict

        len_source = source_data_3dmm['shape'].shape[0]
        len_target = target_data_3dmm['shape'].shape[0]

        source_idx = random.randint(0, len_source-1)
        target_idx = random.randint(0, len_target-1)

        shape_code_new = source_data_3dmm['shape'][source_idx]
        tex_code_new = source_data_3dmm['tex'][source_idx]
        tforms_new = target_data_3dmm['tforms'][target_idx]
        exp_code_new = target_data_3dmm['exp'][target_idx]
        pose_code_new = target_data_3dmm['pose'][target_idx]
        cam_code_new = target_data_3dmm['cam'][target_idx]
        light_code_new = target_data_3dmm['light'][target_idx]

        # get index
        index = (source_idx, target_idx)

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
        source_tensor = get_tensor_clip()(source_image.copy()).to(torch.float16)
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