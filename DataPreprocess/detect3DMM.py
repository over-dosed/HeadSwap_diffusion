## this script is to detect 3dMM

# input : landmarks_dict & numpy images (RGB, 0-1)

import os
import numpy as np
import torch
import pickle
import math
from skimage.transform import estimate_transform, warp
    

def preprocess_for_emoca(image, kpt, crop_size=224, scale=1.25):
    ## process a img to alignment that emoca need

    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])

    old_size = (right - left + bottom - top) / 2 * 1.1
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)
    src_pts = np.array(
        [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2]])
    
    image = image / 255.

    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2, 0, 1)  # h, w, 3 -> 3, h, w
    dst_image = torch.tensor(dst_image).float()
    return dst_image

def detect_3dmm(landmarks_list, aligned_imgs, emoca, save_folder_path = None, crop_size=224, batch_size = 64):
    # aligned_imgs: python list of np.array, (target_size, target_size, 3), RGB, 0-255, uint8

    if isinstance(aligned_imgs, list):

        preprocessed_imgs = torch.zeros((len(aligned_imgs), 3, crop_size, crop_size))
        for i in range(len(aligned_imgs)):
            preprocessed_imgs[i, :] = preprocess_for_emoca(aligned_imgs[i].copy(), landmarks_list[i])
    
    else:
        preprocessed_imgs = preprocess_for_emoca(aligned_imgs.copy(), landmarks_list).view(1, 3, crop_size, crop_size)

    with torch.no_grad():
        steps = int(math.ceil(preprocessed_imgs.shape[0] / batch_size))
        for step in range(steps):
            preprocessed_batch = preprocessed_imgs[step * batch_size : min((step + 1) * batch_size, preprocessed_imgs.shape[0]), :].cuda()
            
            if step == 0:
                shapecode, texcode, expcode, posecode, cam, lightcode = emoca.encode(preprocessed_batch)
            else:
                shapecode_batch, texcode_batch, expcode_batch, posecode_batch, cam_batch, lightcode_batch = emoca.encode(preprocessed_batch)
                shapecode = torch.cat((shapecode, shapecode_batch))
                texcode = torch.cat((texcode, texcode_batch))
                expcode = torch.cat((expcode, expcode_batch))
                posecode = torch.cat((posecode, posecode_batch))
                cam = torch.cat((cam, cam_batch))
                lightcode = torch.cat((lightcode, lightcode_batch))

    dict_3dmm = {}
    dict_3dmm['shapecode'] = shapecode
    dict_3dmm['texcode'] = texcode
    dict_3dmm['expcode'] = expcode
    dict_3dmm['posecode'] = posecode
    dict_3dmm['cam'] = cam
    dict_3dmm['lightcode'] = lightcode

    if save_folder_path is not None:
        with open(os.path.join(save_folder_path, '3DMM.pkl'), 'wb') as f:
            pickle.dump(dict_3dmm, f)


    return dict_3dmm