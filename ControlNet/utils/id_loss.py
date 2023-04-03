import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import cv2

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from utils.arcface_pytorch.models.resnet import resnet_face18

def cosine_distance(u: torch.Tensor, v: torch.Tensor, dim) -> torch.Tensor:
    # 计算两个张量之间的余弦距离
    return 1.0 - F.cosine_similarity(u, v, dim=dim)

class ID_loss(nn.Module):
    def __init__(self, device, arcface_model_path):
            super().__init__()

            self.device = device

            self.face_detector = SFDDetector('cuda')

            self.arcface_model = resnet_face18(False)   # arcface get id information
            state_dict = torch.load(arcface_model_path)
            self.arcface_model.load_state_dict(state_dict)
            self.arcface_model.eval()

            self.arcface_model = self.arcface_model.to(device)

            transformer = torch.nn.Sequential(
                transforms.Resize(size=(128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
            )
            self.transformer = torch.jit.script(transformer)

    def process_a_image(self, bbox, img):
        ##  to process a img for arcface
        ##  img : tensot, float, -1~1, (3, h, w), RGB


        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        center_point = [int((x1 + x2) / 2), int((y1 + y2) / 2)] ## recalculate the center point
        expand_size = int((y2 - y1) * 0.5) # expand_size -- half of the total crop size
        crop_size = expand_size * 2

        new_x1 = center_point[0] - expand_size
        new_y1 = center_point[1] - expand_size

        img = transforms.functional.crop(img, top = new_y1, left = new_x1, height = crop_size, width = crop_size)
        img = self.transformer(img) # resize and to_gray
        img = img.transpose(0, 2) # 128, 128, 1
        img = torch.dstack((img, torch.fliplr(img))) # 128, 128, 2
        img = img.transpose(0, 2) # 2, 128, 128
        img = img.unsqueeze(1) # 2, 1, 128, 128

        img = (img - 127.5) / 127.5

        return img

    def get_id_feature(self, img_batch):
        ## this method get id feature of a image batch
        ## img_batch: tensor,  (B, 3, size, size), RGB, -1~1

        img_batch = (img_batch + 1.0) * 127.5
        img_batch = img_batch.float()
        with torch.no_grad():
            bbox_batch = self.face_detector.detect_from_batch(img_batch)

        preprocessed_batch = []
        for i in range(len(bbox_batch)):
            if len(bbox_batch[i]) == 0:
                # detect no face, return zero image
                zero_image = torch.zeros(2, 1, 128, 128)
                zero_image = zero_image.to(self.device)
                preprocessed_batch.append(zero_image)
            else:
                BBox = bbox_batch[i][0].astype('int32')
                img = img_batch[i]
                img = self.process_a_image(BBox, img)
                preprocessed_batch.append(img)
        preprocessed_batch = torch.concatenate(preprocessed_batch, dim=0).float() # preprocessed_batch : (B*2, 1, 128, 128)

        output_batch = self.arcface_model(preprocessed_batch)
        fe_1 = output_batch[::2]
        fe_2 = output_batch[1::2]
        feature = torch.cat((fe_1, fe_2), dim=1) # (B, 1024)

        return feature

    def forward(self, source_img, output_img):
            
            # 判断source_img的dim，如果 dim ==2 则是feature而不用处理
            if len(source_img.shape) == 2:
                source_id_feature = source_img
            else:
                with torch.no_grad():
                    # source id no need to backprop
                    source_id_feature = self.get_id_feature(source_img) # (B, 1024)

            if len(output_img.shape) == 2:
                ouput_id_feature = output_img
            else:
                ouput_id_feature = self.get_id_feature(output_img) # (B, 1024)
            
            loss = cosine_distance(source_id_feature, ouput_id_feature, dim=1)
            return loss