from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from HSD_dataset import HSD_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = '/data1/wc_log/zxy/control_pbe_ini.ckpt'
batch_size = 2
n_gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

root_path = '/data0/wc_data/VFHQ/train'


if __name__ == "__main__":
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('/home/wenchi/zxy/HSD/ControlNet/models/cldm_pve.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = HSD_Dataset(root_path, batch_size)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=n_gpus, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=n_gpus, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)
