## this script is to get id information with arcface

import os
from PIL import Image
import pickle
import math
import numpy as np

import torch

def process_a_image(lms, img, reshape_size = 128, up_expand_rate = 0.3):
    left = np.min(lms[:, 0])
    right = np.max(lms[:, 0])
    top = np.min(lms[:, 1])
    bottom = np.max(lms[:, 1])

    top = int(max(top - (bottom - top) * up_expand_rate, 0))
    bottom = int(bottom)
    center = [(left + right) * 0.5, (top + bottom) * 0.5]
    half_size = center[1] - top
    left, right = int(center[0] - half_size), int(center[0] + half_size)

    img = Image.fromarray(img[top:bottom, left:right, :]).resize((reshape_size, reshape_size)).convert('L')
    # img.save('/home/zxy/HSD/DataPreprocess/test.jpg')
    img = np.asarray(img)
    img = np.dstack((img, np.fliplr(img)))
    img = img.transpose((2, 0, 1))
    img = img[:, np.newaxis, :, :]
    img = img.astype(np.float32, copy=False)
    img -= 127.5
    img /= 127.5

    # 2 * 1 * 128 * 128

    return img


def get_id_featurs(landmarks_list, aligned_imgs, model, save_folder_path = None, batch_size=64):
    ## aligned_imgs: python list of np.array, (target_size, target_size, 3), RGB, 0-255, uint8
    
    images = None
    for i in range(len(aligned_imgs)):
        image = process_a_image(landmarks_list[i], aligned_imgs[i])

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

    images = torch.from_numpy(images)

    with torch.no_grad():
        steps = int(math.ceil(images.shape[0] / batch_size))
        for step in range(steps):
            imgs_batch = images[step * batch_size : min((step + 1) * batch_size, images.shape[0]), :].cuda()
            
            if step == 0:
                output = model(imgs_batch)
            else:
                output_batch = model(imgs_batch)
                output = torch.cat((output, output_batch))

    output = output.data.cpu().numpy()
    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))

    if save_folder_path is not None:
        with open(os.path.join(save_folder_path, 'id.pkl'), 'wb') as f:
            pickle.dump(feature, f)

    return feature