### this script is to preprocess a floder of images
## filter image size 、 crop&align images 、 detect 3DMM 、 get id information 、 face-parse

import os
import pickle
import numpy as np
from glob import glob
from PIL import Image

from HSD.DataPreprocess.alignImage import process_images
from HSD.DataPreprocess.detect3DMM import detect_3dmm
from HSD.DataPreprocess.faceParse import face_parse
from HSD.DataPreprocess.getIdInformation import get_id_featurs

def preprocessAFolder(args, folder_path, save_folder_path, face_detector, emoca, face_parse_net, arcface_model):

    # get args
    target_size = args.target_size
    expand_rate_up = args.expand_rate_up
    expand_rate_down = args.expand_rate_down

    imagepath_list = sorted(glob(folder_path + '/*.jpg') + glob(folder_path + '/*.png') + glob(folder_path + '/*.bmp'))
    imgs = [Image.open(imagepath).convert("RGB") for imagepath in imagepath_list]

    ## filter image size
    filtered_imgs = [np.asarray(img) for img in imgs if min(img.size) > args.min_origin_size]
    if len(filtered_imgs) < args.min_image_number :
        return

    ## get lms and aligned images
    if os.path.exists(os.path.join(save_folder_path, 'lm.pkl')):
        with open(os.path.join(save_folder_path, 'lm.pkl'), 'rb') as f:
            landmarks_list = pickle.load(f)
        saved_imagepath_list = sorted(glob(save_folder_path + '/*.jpg'))
        aligned_imgs = [np.asarray(Image.open(img_path).convert("RGB")) for img_path in saved_imagepath_list]
    
    else:
        landmarks_list, aligned_imgs = process_images(target_size, expand_rate_up, expand_rate_down, filtered_imgs, face_detector, save_folder_path)

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
    

    ## aligned_imgs: python list of np.array, (target_size, target_size, 3), RGB, 0-255, uint8

    ## get 3DMM
    if not os.path.exists(os.path.join(save_folder_path, '3DMM.pkl')):
        detect_3dmm(landmarks_list, aligned_imgs, emoca, save_folder_path)

    ## get face-parse
    if not os.path.exists(os.path.join(save_folder_path, 'parse.pkl')):
        face_parse(aligned_imgs, face_parse_net, save_folder_path)

    ## get id information
    if not os.path.exists(os.path.join(save_folder_path, 'id.pkl')):
        get_id_featurs(landmarks_list, aligned_imgs, arcface_model, save_folder_path)
        
    return