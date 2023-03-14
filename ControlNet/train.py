from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from HSD_dataset import HSD_Dataset_normal, HSD_Dataset_single
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
# resume_path = '/data1/wc_log/zxy/control_pbe_ini.ckpt'
resume_path =  '/data1/wc_log/zxy/ckpt/v3-epoch=66-global_step=64989.0.ckpt'
log_path = '/home/wenchi/zxy/HSD/image_log/log_v3.1/'
batch_size = 2
n_gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

root_path = '/data0/wc_data/VFHQ/train'


if __name__ == "__main__":
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('/home/wenchi/zxy/HSD/ControlNet/models/cldm_pve_v2.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = HSD_Dataset_normal(root_path)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(log_path, batch_frequency=logger_freq)

    checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="global_step",
    mode="max",
    dirpath="/data1/wc_log/zxy/ckpt/",
    filename="v3.1-{epoch:02d}-{global_step}",
)
    trainer = pl.Trainer(gpus=n_gpus, precision=32, callbacks=[logger, checkpoint_callback])

    # Train!
    trainer.fit(model, dataloader)
