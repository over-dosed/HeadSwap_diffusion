### this script for parse image to get hole head mask

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import pickle

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 0, 0],
                   [0, 0, 0], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [255, 255, 255],
                   [255, 255, 0], [255, 255, 255], [255, 255, 255],
                   [0, 0, 0], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        if len(index[0]) == 0:
            print('not find'+str(pi)+'!  ')
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # org = (index[1][0], index[0][0])
            # fontScale = 0.4
            # color = (0, 0, 0)
            # thickness = 1
            # vis_parsing_anno_color = cv2.putText(vis_parsing_anno_color, str(pi), org, font, fontScale, color, thickness, cv2.LINE_AA)

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def face_parse(aligned_imgs, net, save_folder_path = None):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        imgs_numpy = np.asarray([np.asarray(to_tensor(Image.fromarray(img).resize((512, 512), Image.BILINEAR))) for img in aligned_imgs])
        imgs = torch.from_numpy(imgs_numpy)
        imgs = imgs.cuda()
        out = net(imgs)[0]
        parsing = out.cpu().numpy().argmax(1) # numpy but size is 512 , B * 512 * 512
        parsing = parsing[:, ::2, ::2] # resize to 256

        # only for test
        # image = Image.fromarray(aligned_imgs[50])
        # vis_parsing_maps(image, parsing[50], stride=1, save_im=True, save_path='/home/zxy/HSD/DataPreprocess/test_5.jpg')

    if save_folder_path is not None:
        with open(os.path.join(save_folder_path, 'parse.pkl'), 'wb') as f:
            pickle.dump(parsing, f)

    return parsing
        