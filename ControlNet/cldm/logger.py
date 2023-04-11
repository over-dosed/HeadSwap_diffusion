import os
import cv2
import numpy as np
import torch

import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, save_dir, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
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

    def draw_id_loss(self, loss, grid):
        # grid: (H*B, W, 3), np.uint8, 0~255
        _, W, _ = grid.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        thickness = 2
        fontScale = 1
        fix_up = 30

        grid = grid.copy()
        loss = loss.detach().cpu()
        
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
                N = min(images['d_background'].shape[0], self.max_images)

            # id loss calculate & draw for cross_id
            if split == "train":
                source_image = batch['target']
            else:
                source_image = batch['source_image']

            id_loss_samples = pl_module.ID_loss(source_image[:N].clone(), images['samples'])
            id_loss_cfg = pl_module.ID_loss(source_image[:N].clone(), images['samples_cfg_scale'])

            for k in images:
                images[k] = images[k][:N].float()
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

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
            self.log_img(pl_module, batch, batch_idx, split=batch['flag'][0])