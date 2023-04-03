## this script is to get id information with arcface

import os
import pickle
import math

def get_BBox(imgs_tensor, face_detector, save_folder_path = None, batch_size=64):
    ## imgs_tensor: tensor,  (totalsize, 3, 512, 512), RGB, 0-255, uint8

    steps = int(math.ceil(imgs_tensor.shape[0] / batch_size))
    for step in range(steps):
        imgs_batch = imgs_tensor[step * batch_size : min((step + 1) * batch_size, imgs_tensor.shape[0]), :]
        output_batch = face_detector.detect_from_batch(imgs_batch)
        
        if step == 0:
            output = output_batch
        else:
            output = output + output_batch

    BBox_list = []
    for x in output:
        if len(x) == 0:
            return None
        BBox_list.append(x[0].astype('int32'))

    if save_folder_path is not None:
        with open(os.path.join(save_folder_path, 'BBox.pkl'), 'wb') as f:
            pickle.dump(BBox_list, f)

    return BBox_list