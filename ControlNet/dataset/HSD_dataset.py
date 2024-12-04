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

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)\

def smooth_expand_mask(mask_image, ksize=None, sigmaX= None, sigmaY= None):
    # GaussianBlur to reduce mask edge serrateimport random
    # GaussianBlur to enlarge mask
    if ksize is None or sigmaX is None or sigmaY is None:
        random_int = random.sample(range(-15, 5), 4)
        ksize=(33 + random_int[0]*2, 33 + random_int[1]*2)
        sigmaX= 43 + random_int[2]*2
        sigmaY= 43 + random_int[3]*2
    mask_image = cv2.GaussianBlur(mask_image, ksize, sigmaX=sigmaX, sigmaY = sigmaY)
    mask_image = np.where( (mask_image <= 0), 0, 255).astype('uint8')
    return mask_image

def random_shape_mask(mask_image, random_point_nums = 50):
    enlarged_box = mask_find_bbox(mask_image)

    x_coords = np.random.randint(enlarged_box[0], enlarged_box[2], (random_point_nums, 1))
    y_coords = np.random.randint(enlarged_box[1], enlarged_box[3], (random_point_nums, 1))
    points = np.concatenate([x_coords, y_coords], axis= 1)

    # filter points that out of original mask
    pixel_values = mask_image[x_coords, y_coords]
    mask = np.concatenate([pixel_values == 0, pixel_values == 0], axis= 1)
    black_points = points[mask]
    black_points = np.reshape(black_points, (-1, 2))

    # get the convexHull of points
    hull = cv2.convexHull(black_points)

    try:
        cv2.fillPoly(mask_image, [hull], 255)
    except cv2.error as e:
        pass

    return mask_image

def square_mask(mask_image):
    h, w = mask_image.shape
    mask_square_mask = np.zeros((h, w), np.uint8)
    bbox_target = mask_find_bbox(mask_image)
    mask_square_mask = cv2.rectangle(mask_square_mask, (bbox_target[0], bbox_target[1]), (bbox_target[2], bbox_target[3]), 255,-1)

    return mask_square_mask

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

def render_add_gaze(rendered_image, target_gaze_mask, target_image):
    # rendered_image : (3, 512, 512), tensor, cpu, 0~1, RGB
    # target_gaze_mask: (512, 512), numpy, 0 or 255
    # target_image: (3, 512, 512), numpy, -1~1, RGB

    # get rendered image
    rendered_image = rendered_image.numpy()
    rendered_image_transpose = rendered_image.transpose(1, 2, 0) # (512, 512, 3)

    # get gray target image for gaze image
    target_image_transpose = (target_image.transpose(1, 2, 0) / 2.0) + 0.5 # (512, 512, 3) , 0~1, RGB
    target_image_gray = cv2.cvtColor(target_image_transpose, cv2.COLOR_RGB2GRAY) #(512, 512)
    render_minus_gaze = cv2.bitwise_and(rendered_image_transpose, rendered_image_transpose, mask = 255 - target_gaze_mask)

    # get gaze image
    target_image_gaze = cv2.bitwise_and(target_image_gray, target_image_gray, mask = target_gaze_mask)
    target_image_gaze = np.expand_dims(target_image_gaze, axis=2)
    target_image_gaze = np.repeat(target_image_gaze, 3, axis=2)

    render_add_gaze = render_minus_gaze + target_image_gaze
    render_add_gaze = render_add_gaze.transpose(2, 0, 1) # (3, 512, 512)

    return render_add_gaze

class HSD_Dataset(Dataset):
    # this class is the normal way to get a item of a batch
    def __init__(self, root_path, condition_branch, face_parse_net, face_feature_extractor, flag, lenth = None):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.condition_branch = condition_branch
        self.face_parse_net = face_parse_net
        self.face_feature_extractor = face_feature_extractor
        self.flag = flag
        self.lenth = lenth

    def __len__(self):
        if self.lenth is None:
            return len(self.data)
        else:
            return self.lenth

    def __getitem__(self, idx):
        clip_path = self.data[idx]
        condition_pkl_path = os.path.join(clip_path, '3DMM_condition.pkl')

        with open(condition_pkl_path, 'rb') as f_condition:
            data_3dmm = pickle.load(f_condition)

        codedict, index = self.get_code_dict(data_3dmm)

        # get images paths
        source_image_path = osp.join(clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(clip_path, '{}.png'.format(str(index[1]).zfill(8)))

        # read images & get mask
        source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        source_mask_image, target_mask_image, target_gaze_mask = self.get_mask(source_image, target_image)

        # smooth and enlarge masks
        source_mask_image = smooth_expand_mask(source_mask_image, ksize=(11, 11), sigmaX=11, sigmaY=11)
        target_mask_image = smooth_expand_mask(target_mask_image)
        target_mask_image = random_shape_mask(target_mask_image)

        # process source image
        source_image = cv2.bitwise_and(source_image, source_image, mask = source_mask_image) # get masked
        bbox = mask_find_bbox(source_mask_image)
        source_image = get_align_image(bbox=bbox, img=source_image) # get align & resized source image, (224, 224, 3), numpy, 0~255
        with torch.no_grad():
            id_feature_selected = self.face_feature_extractor(np.expand_dims(source_image, axis=0)).squeeze(0) # get id feature
        source_tensor = get_tensor_clip()(source_image.copy()).to(torch.float16)

        # get masked images (background)
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - target_mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        source_image = (source_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize source images to [-1, 1].
        target_mask_image = np.expand_dims(target_mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        rendered_image = self.condition_branch(codedict).squeeze(0) # 0~1 , 3*512*512, RGB, tensor, cpu
        rendered_image = render_add_gaze(rendered_image, target_gaze_mask, target_image)

        return dict(target=target_image, mask=target_mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, source_image=source_image, hint=rendered_image, flag=self.flag)

    
    def get_mask(self, source_image, target_image):
        # use face parse net to get mask
        source_image_tensor = get_tensor_clip()(source_image.copy())
        target_image_tensor = get_tensor_clip()(target_image.copy())

        input_batch = torch.stack([source_image_tensor, target_image_tensor], dim=0) # (2, 3, 512, 512)

        device = next(self.face_parse_net.parameters()).device
        input_batch = input_batch.to(device)
        out_batch = self.face_parse_net(input_batch)[0].argmax(1).cpu()

        parsing = out_batch.numpy() # numpy but size is 512 , 2 * 512 * 512

        source_parsing = out_batch[0]
        target_parsing = out_batch[1]

        source_mask_image = np.where( (source_parsing == 0)|(source_parsing == 14)|(source_parsing == 16), 0, 255).astype('uint8')  # 0 or 255
        target_mask_image = np.where( (target_parsing == 0)|(target_parsing == 14)|(target_parsing == 16), 0, 255).astype('uint8')  # 0 or 255
        target_gaze_mask = np.where( (target_parsing == 4)|(target_parsing == 5), 255, 0).astype('uint8')

        return source_mask_image, target_mask_image, target_gaze_mask



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

    def __init__(self, root_path, condition_branch, face_parse_net, face_feature_extractor, flag=None, lenth = 5):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.condition_branch = condition_branch
        self.face_parse_net = face_parse_net
        self.face_feature_extractor = face_feature_extractor
        self.flag = flag
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

        with open(source_condition_pkl_path, 'rb') as f_condition_source:
            source_data_3dmm = pickle.load(f_condition_source)
        with open(target_condition_pkl_path, 'rb') as f_condition_target:
            target_data_3dmm = pickle.load(f_condition_target)

        codedict, index = self.get_code_dict(source_data_3dmm, target_data_3dmm)

        # get images paths
        source_image_path = osp.join(source_clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(target_clip_path, '{}.png'.format(str(index[1]).zfill(8)))

        # read images & get mask
        source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        source_mask_image, target_mask_image, target_gaze_mask = self.get_mask(source_image, target_image)

        # smooth and enlarge masks
        source_mask_image = smooth_expand_mask(source_mask_image, ksize=(11, 11), sigmaX=11, sigmaY=11)
        target_mask_image = smooth_expand_mask(target_mask_image, ksize=(77, 77), sigmaX=33, sigmaY=33)

        # process source image
        source_image = cv2.bitwise_and(source_image, source_image, mask = source_mask_image) # get masked
        bbox = mask_find_bbox(source_mask_image)
        source_image = get_align_image(bbox=bbox, img=source_image) # get align & resized source image, (224, 224, 3), numpy, 0~255
        with torch.no_grad():
            id_feature_selected = self.face_feature_extractor(np.expand_dims(source_image, axis=0)).squeeze(0) # get id feature
        source_tensor = get_tensor_clip()(source_image.copy()).to(torch.float16)

        # get masked images (background)
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - target_mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        source_image = (source_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize source images to [-1, 1].
        target_mask_image = np.expand_dims(target_mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        rendered_image = self.condition_branch(codedict).squeeze(0) # (3, h, w)
        rendered_image = render_add_gaze(rendered_image, target_gaze_mask, target_image)

        return dict(target=target_image, mask=target_mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, source_image=source_image, hint=rendered_image, flag=self.flag)

    def get_mask(self, source_image, target_image):
        # use face parse net to get mask
        source_image_tensor = get_tensor_clip()(source_image.copy())
        target_image_tensor = get_tensor_clip()(target_image.copy())

        input_batch = torch.stack([source_image_tensor, target_image_tensor], dim=0) # (2, 3, 512, 512)

        device = next(self.face_parse_net.parameters()).device
        input_batch = input_batch.to(device)
        out_batch = self.face_parse_net(input_batch)[0].argmax(1).cpu()

        parsing = out_batch.numpy() # numpy but size is 512 , 2 * 512 * 512

        source_parsing = out_batch[0]
        target_parsing = out_batch[1]

        source_mask_image = np.where( (source_parsing == 0)|(source_parsing == 14)|(source_parsing == 16), 0, 255).astype('uint8')  # 0 or 255
        target_mask_image = np.where( (target_parsing == 0)|(target_parsing == 14)|(target_parsing == 16), 0, 255).astype('uint8')  # 0 or 255
        target_gaze_mask = np.where( (target_parsing == 4)|(target_parsing == 5), 255, 0).astype('uint8')

        return source_mask_image, target_mask_image, target_gaze_mask
    
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
    def __init__(self, root_path, condition_branch, face_parse_net, face_feature_extractor, flag):
        
        clip_names = sorted(os.listdir(root_path))
        self.data = [os.path.join(root_path, clip_name) for clip_name in clip_names]
        self.condition_branch = condition_branch
        self.face_parse_net = face_parse_net
        self.face_feature_extractor = face_feature_extractor
        self.flag = flag
        self.lenth = 4800

    def __len__(self):
        return self.lenth

    def __getitem__(self, idx):
        clip_path = self.data[0] # for single finetune
        condition_pkl_path = os.path.join(clip_path, '3DMM_condition.pkl')

        with open(condition_pkl_path, 'rb') as f_condition:
            data_3dmm = pickle.load(f_condition)

        codedict, index = self.get_code_dict(data_3dmm)

        # get images paths
        source_image_path = osp.join(clip_path, '{}.png'.format(str(index[0]).zfill(8)))
        target_image_path = osp.join(clip_path, '{}.png'.format(str(index[1]).zfill(8)))

        # read images & get mask
        source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
        target_image = np.asarray(Image.open(target_image_path).convert("RGB"))
        source_mask_image, target_mask_image, target_gaze_mask = self.get_mask(source_image, target_image)

        # smooth and enlarge masks
        source_mask_image = smooth_expand_mask(source_mask_image, ksize=(11, 11), sigmaX=11, sigmaY=11)
        target_mask_image = smooth_expand_mask(target_mask_image)
        target_mask_image = square_mask(target_mask_image)

        # process source image
        source_image = cv2.bitwise_and(source_image, source_image, mask = source_mask_image) # get masked
        bbox = mask_find_bbox(source_mask_image)
        source_image = get_align_image(bbox=bbox, img=source_image) # get align & resized source image, (224, 224, 3), numpy, 0~255
        with torch.no_grad():
            id_feature_selected = self.face_feature_extractor(np.expand_dims(source_image, axis=0)).squeeze(0) # get id feature
        source_tensor = get_tensor_clip()(source_image.copy()).to(torch.float16)

        # get masked images (background)
        bg_image = cv2.bitwise_and(target_image, target_image, mask = 255 - target_mask_image)

        target_image = (target_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize target images to [-1, 1].
        source_image = (source_image.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # Normalize source images to [-1, 1].
        target_mask_image = np.expand_dims(target_mask_image.astype(np.float32) / 255.0, axis=0)

        bg_image = bg_image.astype(np.float32) / 255.0
        bg_image = torch.from_numpy(bg_image.transpose(2, 0, 1))

        rendered_image = self.condition_branch(codedict).squeeze(0) # 0~1 , 3*512*512, RGB, tensor, cpu
        rendered_image = render_add_gaze(rendered_image, target_gaze_mask, target_image)

        return dict(target=target_image, mask=target_mask_image, background=bg_image, source_global=source_tensor, source_id=id_feature_selected, source_image=source_image, hint=rendered_image, flag=self.flag)

    
    def get_mask(self, source_image, target_image):
        # use face parse net to get mask
        source_image_tensor = get_tensor_clip()(source_image.copy())
        target_image_tensor = get_tensor_clip()(target_image.copy())

        input_batch = torch.stack([source_image_tensor, target_image_tensor], dim=0) # (2, 3, 512, 512)
        
        device = next(self.face_parse_net.parameters()).device
        input_batch = input_batch.to(device)
        out_batch = self.face_parse_net(input_batch)[0].argmax(1).cpu()

        parsing = out_batch.numpy() # numpy but size is 512 , 2 * 512 * 512

        source_parsing = out_batch[0]
        target_parsing = out_batch[1]

        source_mask_image = np.where( (source_parsing == 0)|(source_parsing == 14)|(source_parsing == 16), 0, 255).astype('uint8')  # 0 or 255
        target_mask_image = np.where( (target_parsing == 0)|(target_parsing == 14)|(target_parsing == 16), 0, 255).astype('uint8')  # 0 or 255
        target_gaze_mask = np.where( (target_parsing == 4)|(target_parsing == 5), 255, 0).astype('uint8')

        return source_mask_image, target_mask_image, target_gaze_mask



    def get_code_dict(self, code_dict, total_num = 32, pose_threshold = 0.02, loop_max_times = 20):
        # this method get original a clip code_dict as input
        # return the indexs selected randomly and the corresponding combined code_dict

        tforms = code_dict['tforms']
        shape_code = code_dict['shape']
        tex_code = code_dict['tex']
        exp_code = code_dict['exp']
        pose_code = code_dict['pose']
        cam_code = code_dict['cam']
        light_code = code_dict['light']

        total_num = total_num
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

