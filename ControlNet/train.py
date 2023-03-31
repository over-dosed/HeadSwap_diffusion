from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset.HSD_dataset import HSD_Dataset, HSD_Dataset_cross
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
model_name = 'v3.6'

resume_path =  '/data1/wc_log/zxy/ckpt/v3.6-begin.ckpt'
model_cofig_path = '/home/wenchi/zxy/HSD/ControlNet/models/cldm_pve_v3.6.yaml'
ckpt_save_path = "/data1/wc_log/zxy/ckpt/"
root_path = '/data0/wc_data/VFHQ/train'
cross_root_path = '/data0/wc_data/VFHQ/test'

batch_size = 3
n_gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

log_path = '/home/wenchi/zxy/HSD/image_log/log_{}/'.format(model_name)

# arcface model path
arcface_model_path = '/home/wenchi/zxy/HSD/ControlNet/utils/arcface_pytorch/checkpoints/resnet18_110_onecard.pth'


if __name__ == "__main__":
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_cofig_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # dataset & dataloader
    dataset = HSD_Dataset(root_path)
    cross_eval_dataset = HSD_Dataset_cross(root_path, flag='eval', lenth = 5)
    cross_test_dataset = HSD_Dataset_cross(cross_root_path, flag='test', lenth = 5)

    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)
    cross_eval_dataloader = DataLoader(cross_eval_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)
    cross_test_dataloader = DataLoader(cross_test_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)

    # callbacks
    logger = ImageLogger(log_path, batch_frequency=logger_freq, arcface_model_path=arcface_model_path, ddim_steps=50)

    checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="global_step",
    mode="max",
    dirpath=ckpt_save_path,
    filename=model_name + "-{epoch:02d}-{global_step}",
)

    trainer = pl.Trainer(gpus=n_gpus, precision=32, callbacks=[logger, checkpoint_callback])

    # Train!
    trainer.fit(model, dataloader, [cross_eval_dataloader, cross_test_dataloader])
