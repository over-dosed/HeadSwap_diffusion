## this script is to detect 3dMM

# input : landmarks_dict & numpy images (RGB, 0-1)
import cv2
import os
import numpy as np
import torch
import pickle
import math
from skimage.transform import estimate_transform, warp
from PIL import Image
    

def preprocess_for_DECA(image, bbox, crop_size=224, scale=1.25):
    ## process a img to alignment that DECA need
    # image : 3, 512, 512

    left = bbox[0]
    right = bbox[2]
    top = bbox[1]
    bottom = bbox[3]

    old_size = (right - left + bottom - top)/2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])

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

    tform = torch.tensor(tform.params).float()
    tform = torch.inverse(tform).transpose(0, 1)
    
    return dst_image, tform

def detect_3dmm(bboxlist, imgs_numpy, deca, save_folder_path = None, crop_size=224, batch_size = 64):
    # imgs_numpy: numpy  (total_size, 3, 512, 512), RGB, 0-255, uint8

    # need to save for each frame (all tensor): 
    # 1. tforms                                                                                                 [ preprocess output ]
    # 2. shape, tex, exp, pose, cam, light                                                                      [ encode output ]
    # 3. verts, trans_verts (512 size), landmarks2d (512 size), landmarks3d (512 size)                          [ decode output ]

    preprocessed_imgs = torch.zeros((imgs_numpy.shape[0], 3, crop_size, crop_size))
    tforms = torch.zeros((imgs_numpy.shape[0], 3, 3))
    for i in range(imgs_numpy.shape[0]):
        preprocessed_imgs[i, :], tforms[i, :] = preprocess_for_DECA(imgs_numpy[i].copy().transpose(1, 2, 0), bboxlist[i])

    dict_3DMM = {}
    condition_dict_3DMM = {}
    feature_dict_3DMM = {}
    dict_3DMM['tforms'] = tforms
    condition_keys = ['tforms', 'shape', 'tex', 'exp', 'pose', 'cam', 'light']
    feature_keys = ['trans_verts']

    with torch.no_grad():
        steps = int(math.ceil(preprocessed_imgs.shape[0] / batch_size))
        for step in range(steps):
            preprocessed_batch = preprocessed_imgs[step * batch_size : min((step + 1) * batch_size, preprocessed_imgs.shape[0]), :].cuda()
            tforms_batch = tforms[step * batch_size : min((step + 1) * batch_size, tforms.shape[0]), :].cuda()
            
            codedict = deca.encode(preprocessed_batch, use_detail = False)
            opdict = deca.fast_decode(codedict, tform=tforms_batch)

            # for landmarks:
            # predicted_landmark[...,0] = predicted_landmark[...,0]*image.shape[1]/2 + image.shape[1]/2
            # predicted_landmark[...,1] = predicted_landmark[...,1]*image.shape[0]/2 + image.shape[0]/2

            # only for test
            # reder_image = deca.render_for_hsd(codedict, original_image=original_images_batch, tform=tforms_batch)         render code
            # render_image = Image.fromarray((reder_image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))
            # render_image.save('/home/wenchi/zxy/HSD/test.jpg')
            
            if step == 0:
                for key in codedict:
                    dict_3DMM[key] = codedict[key].cpu()
                for key in opdict:
                    dict_3DMM[key] = opdict[key].cpu()

            else:
                for key in codedict:
                    dict_3DMM[key] = torch.cat((dict_3DMM[key], codedict[key].cpu()))
                for key in opdict:
                    dict_3DMM[key] = torch.cat((dict_3DMM[key], opdict[key].cpu()))

    # print(' ')
    # for key in dict_3DMM:
    #     print(str(key) + '  , shape : ' + str(dict_3DMM[key].shape))

    condition_dict_3DMM = {key:dict_3DMM[key] for key in condition_keys}
    feature_dict_3DMM = {key:dict_3DMM[key] for key in feature_keys}

    if save_folder_path is not None:
        with open(os.path.join(save_folder_path, '3DMM.pkl'), 'wb') as f:
            pickle.dump(dict_3DMM, f)
        with open(os.path.join(save_folder_path, '3DMM_condition.pkl'), 'wb') as f:
            pickle.dump(condition_dict_3DMM, f)
        with open(os.path.join(save_folder_path, '3DMM_feature.pkl'), 'wb') as f:
            pickle.dump(feature_dict_3DMM, f)

    return dict_3DMM