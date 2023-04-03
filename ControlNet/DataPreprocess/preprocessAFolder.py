### this script is to preprocess a floder of images
## filter image size 、 crop&align images 、 detect 3DMM 、 get id information 、 face-parse

import os
import pickle
import numpy as np
from glob import glob
from PIL import Image
import shutil

import torch

from HSD.DataPreprocess.getBBox import get_BBox
from HSD.DataPreprocess.detect3DMM import detect_3dmm
from HSD.DataPreprocess.faceParse import face_parse
from HSD.DataPreprocess.getIdInformation import get_id_featurs

def preprocessAFolder(args, folder_path, face_detector, deca, face_parse_net, arcface_model):

    # get args
    # target_size = args.target_size
    # expand_rate_up = args.expand_rate_up
    # expand_rate_down = args.expand_rate_down

    imagepath_list = sorted(glob(folder_path + '/*.png'))
    imgs_numpy_list = [np.asarray(Image.open(imagepath).convert("RGB")) for imagepath in imagepath_list]

    ## imgs_numpy_list: python list of np.array, (512, 512, 3), RGB, 0-255, uint8

    ## filter image size
    # filtered_imgs = [np.asarray(img) for img in imgs if min(img.size) > args.min_origin_size]
    if len(imgs_numpy_list) < args.min_image_number :
        return

    ## imgs_numpy: numpy, (totalsize, 3, 512, 512), RGB, 0-255, uint8
    imgs_numpy = np.asarray(imgs_numpy_list).transpose(0, 3, 1, 2)
    imgs_tensor = torch.from_numpy(imgs_numpy)
    batch_size = 64

    ## get BBox
    if os.path.exists(os.path.join(folder_path, 'BBox.pkl')):
        with open(os.path.join(folder_path, 'BBox.pkl'), 'rb') as f:
            bboxlist = pickle.load(f)
    else:   
        bboxlist = get_BBox(imgs_tensor, face_detector, folder_path, batch_size)
        if bboxlist is None:
            shutil.rmtree(folder_path)
            print('delete' + str(folder_path))
            return
    # bboxlist : list of bbox [x1, y1, x2, y2]

    # only for test ; test lms
    # test_img = aligned_imgs[31]
    # landmarks = landmarks_list[31]
    # test_img = Image.fromarray(test_img)
    # for lm in landmarks:
    #     test_img.putpixel((int(lm[0]-1), int(lm[1]-1)), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]-1), int(lm[1])), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]-1), int(lm[1]+1)), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]), int(lm[1]-1)), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]), int(lm[1])), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]), int(lm[1]+1)), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]+1), int(lm[1]-1)), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]+1), int(lm[1])), (255, 0, 0))
    #     test_img.putpixel((int(lm[0]+1), int(lm[1]+1)), (255, 0, 0))
    # test_img.save("/home/zxy/HSD/DataPreprocess/test.jpg")

    ## get face-parse
    if not os.path.exists(os.path.join(folder_path, 'mask_00000001.jpg')):
        face_parse(imgs_numpy, face_parse_net, folder_path, batch_size = batch_size)

    ## get 3DMM
    if not os.path.exists(os.path.join(folder_path, '3DMM.pkl')):
        detect_3dmm(bboxlist, imgs_numpy, deca, folder_path, batch_size = batch_size)

    ## get id information
    if not os.path.exists(os.path.join(folder_path, 'id.pkl')):
        get_id_featurs(bboxlist, imgs_numpy, arcface_model, folder_path, batch_size = batch_size)
        
    return