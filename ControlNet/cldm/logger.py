import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from utils.arcface_pytorch.models.resnet import resnet_face18


def cosine_distance(u: torch.Tensor, v: torch.Tensor, dim) -> torch.Tensor:
    # 计算两个张量之间的余弦距离
    return 1.0 - F.cosine_similarity(u, v, dim=dim)


class ImageLogger(Callback):
    def __init__(self, save_dir, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, arcface_model_path=None,
                 **log_images_kwargs):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.save_dir = save_dir

        self.face_detector = SFDDetector(device='cpu')

        self.arcface_model = resnet_face18(False)   # arcface get id information
        state_dict = torch.load(arcface_model_path)
        self.arcface_model.load_state_dict(state_dict)
        self.arcface_model.eval()

    def process_a_image(self, bbox, img, reshape_size = 128):
        ##  to process a img for arcface
        ##  img : numpy, uint8, 0~255, (3, h, w), RGB
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
        img = np.asarray(aligned_img)
        img = np.dstack((img, np.fliplr(img)))
        img = img.transpose((2, 0, 1))
        img = img[:, np.newaxis, :, :]
        img = img.astype(np.float32, copy=False)
        img -= 127.5
        img /= 127.5

        # 2 * 1 * 128 * 128

        return img

    @torch.no_grad()
    def get_id_feature(self, img_batch):
        ## this method get id feature of a image batch
        ## img_batch: tensor,  (B, 3, size, size), RGB, -1~1

        img_batch = (img_batch + 1.0) * 127.5

        output_batch = self.face_detector.detect_from_batch(img_batch)

        preprocessed_batch = []
        for i in range(len(output_batch)):
            if len(output_batch[i]) == 0:
                # detect no face, return zero image
                preprocessed_batch.append(np.zeros((2, 1, 128, 128)).astype(np.float32))
            else:
                BBox = output_batch[i][0].astype('int32')
                img = img_batch[i].numpy().astype('uint8')
                img = self.process_a_image(BBox, img)
                preprocessed_batch.append(img)
        preprocessed_batch = np.concatenate(preprocessed_batch, axis=0) # preprocessed_batch : (B*2, 1, 128, 128)
        preprocessed_batch = torch.from_numpy(preprocessed_batch).float()

        output_batch = self.arcface_model(preprocessed_batch)
        fe_1 = output_batch[::2]
        fe_2 = output_batch[1::2]
        feature = torch.cat((fe_1, fe_2), dim=1)

        return feature
    
    @torch.no_grad()   
    def get_id_loss(self, source_img, output_img):
        source_id_feature = self.get_id_feature(source_img)
        ouput_id_feature = self.get_id_feature(output_img) # (B, 1024)

        loss = cosine_distance(source_id_feature, ouput_id_feature, dim=1)
        return loss

    def draw_id_loss(self, loss, grid):
        # grid: (H*B, W, 3), np.uint8, 0~255
        _, W, _ = grid.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        thickness = 2
        fontScale = 1
        fix_up = 30

        grid = grid.copy()
        for i in range(loss.shape[0]):
            text = str(np.around(loss[i].numpy(), decimals=4))
            x = 0
            y = W * i + fix_up
            cv2.putText(grid, text, (x,y), font, fontScale=fontScale, color=color, thickness=thickness)
        
        return grid

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx, 
                  img_size = 512, id_loss_samples=None, id_loss_cfg=None):
        save_dir = self.save_dir
        root = os.path.join(save_dir, "image_log", split)
        keys = sorted(images.keys())
        grid_list = []
        for k in keys:
            B = images[k].shape[0]
            grid = torchvision.utils.make_grid(images[k], nrow=1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8) # H*B, W, 3
            if images[k].shape[2] != img_size:
                if B == 1:
                    grid = np.asarray(Image.fromarray(grid).resize((img_size, img_size)))
                else:
                    grid = np.asarray(Image.fromarray(grid).resize((img_size + 4, (img_size * B + 2 * (B + 1)))))

            # draw id loss
            if k == 'samples' and id_loss_samples is not None:
                grid = self.draw_id_loss(id_loss_samples, grid)

            if k == 'samples_cfg_scale' and id_loss_cfg is not None:
                grid = self.draw_id_loss(id_loss_cfg, grid)

            grid_list.append(grid)

        full_grid = np.concatenate(grid_list, axis=1)

        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(full_grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                images['d_background'] = batch['background']

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N].float()
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # id loss calculate & draw for cross_id
            source_image = batch['source_image'].detach().cpu()

            id_loss_samples = self.get_id_loss(source_image[:N].clone(), images['samples'])
            id_loss_cfg = self.get_id_loss(source_image[:N].clone(), images['samples_cfg_scale'])

            if split == 'train':
                id_loss_dict = {'id_samples_train':id_loss_samples.mean(), 'id_samples_cfg_train':id_loss_cfg.mean()}
            else:
                id_loss_dict = {'id_samples_' + batch['flag'][0]:id_loss_samples.mean(), 'id_samples_cfg_' + batch['flag'][0]:id_loss_cfg.mean()}
            pl_module.self_log_dict(id_loss_dict)

            self.log_local(split, images, pl_module.global_step, pl_module.current_epoch, batch_idx, 
                            id_loss_samples=id_loss_samples, id_loss_cfg=id_loss_cfg)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx, split):
        if split == 'train':
            return check_idx % self.batch_freq == 0
        else:
            return True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="cross_id_" + batch['flag'][0])