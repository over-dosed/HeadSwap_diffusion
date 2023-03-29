### this script is to detect landmarks and crop&align images

import os
import shutil
import numpy as np
import pickle
import torch
from PIL import Image

def process_a_image(original_img, face_detector, target_size, expand_rate_up, expand_rate_down):

    h, w, _ = original_img.shape
    
    with torch.no_grad():
        bbox, landmarks = face_detector.run(original_img)

    if bbox is None:  # detect no face
        return None, None

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    center_point = [int((x1 + x2) / 2), int((expand_rate_down - expand_rate_up + 1) / 2 * (y2 - y1) + y1 )] ## recalculate the center point

    expand_size = int((y2 - y1) * 0.5 * (expand_rate_up + expand_rate_down + 1)) # expand_size -- half of the total crop size
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
    aligned_img[crop_top:crop_bottom, crop_left:crop_right] = original_img[origin_top:origin_bottom, origin_left:origin_right]
    aligned_img = Image.fromarray(aligned_img)
    aligned_img = aligned_img.resize((target_size, target_size), Image.LANCZOS)
    aligned_img = np.asarray(aligned_img)

    landmarks = transform_lms(landmarks, new_x1, new_y1, crop_size, target_size)
    return landmarks, aligned_img

def transform_lms(lms, new_x1, new_y1, crop_size, target_size):

    for lm in lms:
        lm[0] = int((lm[0] - new_x1) / crop_size * target_size)
        lm[1] = int((lm[1] - new_y1) / crop_size * target_size)

    return lms


def process_images(target_size, expand_rate_up, expand_rate_down, filtered_imgs, face_detector, save_folder_path = None):
    # filtered_imgs : python list of numpy images (RGB, 0~255, random size, uint8)

    
    landmarks_list = []
    aligned_imgs = []

    for original_img in filtered_imgs:
        landmarks, aligned_img = process_a_image(original_img, face_detector, target_size, expand_rate_up, expand_rate_down)
        # landmarks: (68,2) , (x, y)
        # aligned_img: np.array, (target_size, target_size, 3), RGB, 0-255, uint8

        if landmarks is None:  # detect no face
            continue

        aligned_imgs.append(aligned_img)
        landmarks_list.append(landmarks)

    if save_folder_path is not None:
        saveData(landmarks_list, aligned_imgs, save_folder_path)

    return landmarks_list, aligned_imgs

def saveData(landmarks_list, aligned_imgs, save_folder_path):
    
    if os.path.exists(save_folder_path):
        shutil.rmtree(save_folder_path)
    os.mkdir(save_folder_path)

    for i in range(len(aligned_imgs)):
        image = aligned_imgs[i]
        image = Image.fromarray(image)
        save_img_name = str(i).zfill(5) + '.jpg'
        image.save(os.path.join(save_folder_path, save_img_name))

    with open(os.path.join(save_folder_path, 'lm.pkl'), 'wb') as f:
        pickle.dump(landmarks_list, f)
    
    return