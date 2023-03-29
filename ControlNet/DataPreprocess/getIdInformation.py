## this script is to get id information with arcface

import os
from PIL import Image
import pickle
import math
import numpy as np

import torch

def process_a_image(bbox, img, reshape_size = 128):

    img = img.transpose(1, 2, 0) # 3, h, w -> h, w, 3
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
    aligned_img = aligned_img.resize((reshape_size, reshape_size), Image.LANCZOS).convert('L')
    # aligned_img.save('/home/wenchi/zxy/HSD/test_arcface.jpg')
    img = np.asarray(aligned_img)
    img = np.dstack((img, np.fliplr(img)))
    img = img.transpose((2, 0, 1))
    img = img[:, np.newaxis, :, :]
    img = img.astype(np.float32, copy=False)
    img -= 127.5
    img /= 127.5

    # 2 * 1 * 128 * 128

    return img


def get_id_featurs(bboxlist, aligned_imgs, model, save_folder_path = None, batch_size=64):
    ## aligned_imgs: python list of np.array, (B, 3, target_size, target_size), RGB, 0-255, uint8
    
    images = None
    for i in range(len(aligned_imgs)):
        image = process_a_image(bboxlist[i], aligned_imgs[i])

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

    images = torch.from_numpy(images)

    with torch.no_grad():
        steps = int(math.ceil(images.shape[0] / batch_size))
        for step in range(steps):
            imgs_batch = images[step * batch_size : min((step + 1) * batch_size, images.shape[0]), :].cuda()
            output_batch = model(imgs_batch).cpu()
            
            if step == 0:
                output = output_batch
            else:
                output = torch.cat((output, output_batch))

    output = output.data.numpy()
    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))

    if save_folder_path is not None:
        with open(os.path.join(save_folder_path, 'id.pkl'), 'wb') as f:
            pickle.dump(feature, f)

    return feature