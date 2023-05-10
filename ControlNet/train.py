from share import *

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset.HSD_dataset import HSD_Dataset, HSD_Dataset_cross, HSD_Dataset_single

# condtion Branch & face parse & face feature extractor for condition
from modules.ConditionBranch import Condition_Branch
from modules.Face_feature import FaceFeatureExtractor
from utils.face_parse.model import BiSeNet

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
model_name = 'v3.7_adapter'

# resume_path =  '/data1/wc_log/zxy/ckpt/v3.6.1-epoch=46-global_step=1926.0.ckpt'
resume_path =  '/data1/wc_log/zxy/ckpt/v3.7_adapter-begin.ckpt'
model_cofig_path = '/home/wenchi/zxy/HSD/ControlNet/models/cldm_pve_v3.7.yaml'
ckpt_save_path = "/data1/wc_log/zxy/ckpt/"
# root_path = '/data1/wc_log/zxy/custom_dataset/jjk_video_1' # for single id train
root_path = '/data1/wc_log/zxy/VFHQ/train' # for single id train
cross_root_path = '/data1/wc_log/zxy/VFHQ/test'

batch_size = 4
accumulate_grad_batches = 8
n_gpus = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
control_locked = False
only_mid_control = False

log_path = '/data1/wc_log/zxy/image_log/log_{}/'.format(model_name)


if __name__ == "__main__":
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_cofig_path).cpu()

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.control_locked = control_locked
    model.only_mid_control = only_mid_control


    # dataset & dataloader

    ## condition branch
    condition_Branch = Condition_Branch(device='cuda:1')

    ## face parse
    face_parse_net = BiSeNet(n_classes=19)
    save_pth = '/home/wenchi/zxy/HSD/ControlNet/utils/face_parse/res/cp/79999_iter.pth'
    face_parse_net.load_state_dict(torch.load(save_pth))
    face_parse_net.cuda()
    face_parse_net.eval()

    ## face feature extractor
    face_feature_net = FaceFeatureExtractor('/home/wenchi/zxy/HSD/ControlNet/utils/Face_Alignment/retinaface/weights/mobilenet0.25_Final.pth', 
                                            '/home/wenchi/zxy/HSD/ControlNet/utils/arcface/model_data/arcface_mobilenet_v1.pth', 
                                            output_size=(112, 112), device = 'cuda:1')


    dataset = HSD_Dataset(root_path, condition_Branch, face_parse_net, face_feature_net, flag='train')
    same_test_dataset = HSD_Dataset(cross_root_path, condition_Branch, face_parse_net, face_feature_net, flag='same_test', lenth = batch_size)
    cross_test_dataset = HSD_Dataset_cross(cross_root_path, condition_Branch, face_parse_net, face_feature_net, flag='cross_test', lenth = batch_size)

    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)
    cross_eval_dataloader = DataLoader(same_test_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)
    cross_test_dataloader = DataLoader(cross_test_dataset, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)

    # callbacks
    logger = ImageLogger(log_path, batch_frequency=logger_freq, ddim_steps=50)

    checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="global_step",
    mode="max",
    dirpath=ckpt_save_path,
    filename=model_name + "-{epoch:02d}-{global_step}",
)

    # trainer = pl.Trainer(gpus=n_gpus, precision=32, callbacks=[logger, checkpoint_callback])
    trainer = pl.Trainer(gpus=n_gpus, precision=16, callbacks=[logger, checkpoint_callback], accumulate_grad_batches=accumulate_grad_batches)

    # Train!
    trainer.fit(model, dataloader, [cross_eval_dataloader, cross_test_dataloader])
