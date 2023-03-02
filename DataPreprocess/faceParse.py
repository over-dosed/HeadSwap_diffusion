### this script for parse image to get hole head mask

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import math
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

def face_parse(imgs_numpy, net, save_folder_path = None, batch_size = 64):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        imgs_tensor = torch.zeros((imgs_numpy.shape[0], 3, 512, 512))
        for i in range(imgs_numpy.shape[0]):
            temp = to_tensor(imgs_numpy[i].transpose(1, 2, 0))
            imgs_tensor[i, :] = temp

        steps = int(math.ceil(imgs_tensor.shape[0] / batch_size))
        for step in range(steps):
            imgs_batch = imgs_tensor[step * batch_size : min((step + 1) * batch_size, imgs_tensor.shape[0]), :].cuda()
            out_batch = net(imgs_batch)[0].argmax(1).cpu()

            if step == 0:
                out = out_batch
            else:
                out = torch.cat((out, out_batch))

    parsing = out.numpy() # numpy but size is 512 , B * 512 * 512

    masks = np.where( (parsing == 0)|(parsing == 14)|(parsing == 16), 0, 255).astype('uint8')  # 0 or 255
    for i in range(masks.shape[0]):
        mask = cv2.GaussianBlur(masks[i], (15, 15), 15)
        mask = np.where( (mask <= 0), 0, 255).astype('uint8')  # 0 or 255
        if save_folder_path is not None:
            cv2.imwrite(os.path.join(save_folder_path, 'mask_'+str(i).zfill(8)+'.jpg'), mask)

    return parsing
        