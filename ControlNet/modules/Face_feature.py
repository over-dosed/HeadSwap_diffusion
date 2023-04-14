# this file is to extract the face feature from the face image


import torch
import torch.nn as nn
import numpy as np

from utils.Face_Alignment.align_faces import warp_and_crop_face_tensor, get_reference_facial_points
from utils.Face_Alignment.retinaface.detector import RetinafaceDetector
from utils.arcface.nets.arcface import Arcface as arcface

class FaceFeatureExtractor(nn.Module):
    def __init__(self, retina_path, arcface_model_path, output_size=(112, 112), device = 'cuda'):
        '''
        Initializes the FaceFeatureExtractor class.

        Args:
            output_size (Tuple[int, int]): The output size of the aligned face image. Default is (112, 112).
        '''
        super(FaceFeatureExtractor, self).__init__()
        
        # Initializes the RetinafaceDetector class
        self.device = device
        self.detector = RetinafaceDetector(retina_path, type=self.device)
        self.extractor = self.init_arcface(arcface_model_path)
        self.output_size = output_size

    def init_arcface(self, model_path):
        '''
        Initializes the arcface model.

        Args:
            model_path (str): The path to the arcface model.

        Returns:
            arcface_model (Arcface): The initialized arcface model.
        '''
        # Initializes the arcface model
        arcface_model = arcface(backbone='mobilenetv1', mode="predict").eval()
        arcface_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        arcface_model = arcface_model.to(self.device)
        return arcface_model

    def crop_align_image(self, img) -> torch.Tensor:
        '''
        Crop and align a face image.

        Args:
            img: The input image tensor or numpy, with shape (3, size, size) / (size, size, 3) and values in the range of 0~255.

        Returns:
            torch.Tensor: The aligned face image tensor, with shape (1, 3, 112, 112) and values in the range of 0~255.
        '''
        if isinstance(img, torch.Tensor):
            img = img.unsqueeze(0)
        else:
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1).unsqueeze(0)
        
        _, facial5points = self.detector.detect_faces(img)

        # detect no face
        if len(facial5points) == 0:
            return torch.zeros((1, 3, 112, 112))

        facial5points = np.reshape(facial5points[0], (2, 5))

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            self.output_size, inner_padding_factor, outer_padding, default_square)

        # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
        dst_img = warp_and_crop_face_tensor(img, facial5points, reference_pts=reference_5pts, crop_size=self.output_size) # tensor, (1, 3, 112, 112), 0~1, RGB
        dst_img = dst_img.to(self.device)
        return dst_img
    
    # use extractor to extract the face feature
    def extract_feature(self, img: torch.Tensor) -> torch.Tensor:
        '''
        Extract the face feature from the aligned face image.

        Args:
            img (torch.Tensor): The aligned face image tensor, with shape (B, 3, 112, 112) and values in the range of -1~1.

        Returns:
            torch.Tensor: The face feature tensor, with shape (B, size).
        '''
        img = img.to(self.device)
        feature = self.extractor(img)
        return feature
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        '''
        Crop and align a batch of face image, and then extract the face feature.

        Args:
            img (torch.Tensor): The input image tensor or numpy, with shape (B, 3, size, size) and values in the range of 0~255.

        Returns:
            torch.Tensor: The face feature tensor, with shape (B, 512).
        '''
        # loop through each image in the batch and crop and align the face
        croped_img_list = []
        for i in range(img.shape[0]):
            croped_img = self.crop_align_image(img[i]).to(self.device)
            croped_img_list.append(croped_img)  # (1, 3, 112, 112), 0~255
        croped_img = torch.cat(croped_img_list, dim=0)  # (B, 3, 112, 112), 0~255

        croped_img = (croped_img - 127.5) / 127.5 # (B, 3, 112, 112), -1~1

        feature = self.extract_feature(croped_img) # (B, 512)
        return feature