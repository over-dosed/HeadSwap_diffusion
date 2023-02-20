### this script is to preprocess the hole vggface2 train/test dataset

## vggface2/
## train/  test/
## n ****/
## 00**_0*.jpg

import os
import sys
import argparse
from tqdm import tqdm

import torch

from HSD.DataPreprocess.preprocessAFolder import preprocessAFolder
from HSD.DataPreprocess.FaceDetector import FAN
from HSD.utils.emoca.load import load_model
from HSD.utils.face_parse.model import BiSeNet
from HSD.utils.arcface_pytorch.models.resnet import resnet_face18

def main():
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    parser.add_argument('--root_path', type=str, help="vggface2 train or test path. e.g. /home/zxy/data/vggface2/train" )
    parser.add_argument('--save_root_path', type=str, help="the root path to save preprocessed data. e.g. /home/zxy/data/vggface2/train" )
    parser.add_argument('--start_id', type=str, default=None, help="the id you want to start from")

    # filter image
    parser.add_argument('--min_origin_size', type=int, default=256)
    parser.add_argument('--min_image_number', type=int, default=15)

    # crop@align
    parser.add_argument('--target_size', type=int, default=256, help="size if aligned images")
    parser.add_argument('--expand_rate_up', type=float, default=1, help="up expand rate for croping images")
    parser.add_argument('--expand_rate_down', type=float, default=0.3, help="down expand rate for croping images")

    # detect 3DMM
    parser.add_argument('--emoca_asset_dir', type=str, default='/home/wenchi/zxy/HSD/utils/emoca/assets', help="emoca asset dir")

    # face parse
    parser.add_argument('--face_parse_dir', type=str, default='/home/wenchi/zxy/HSD/utils/face_parse/res/cp/79999_iter.pth', help="face parse cp dir")

    # id feature
    parser.add_argument('--arcface_model_path', type=str, default='/home/wenchi/zxy/HSD/utils/arcface_pytorch/checkpoints/resnet18_110_onecard.pth', help="face parse cp dir")
    
    args = parser.parse_args()

    ### prepare models
    face_detector = FAN()   # face detect

    asset_dir = args.emoca_asset_dir         # detect 3DMM
    path_to_models = os.path.join(asset_dir, 'EMOCA', 'models')
    emoca, conf = load_model(path_to_models, asset_dir, 'EMOCA', 'detail')
    emoca.cuda()
    emoca.eval()

    n_classes = 19                          # face parse
    face_parse_net = BiSeNet(n_classes=n_classes)
    save_pth = args.face_parse_dir
    face_parse_net.load_state_dict(torch.load(save_pth))
    face_parse_net.cuda()
    face_parse_net.eval()

    arcface_model = resnet_face18(False)   # arcface get id information
    state_dict = torch.load(args.arcface_model_path)
    arcface_model.load_state_dict(state_dict)
    arcface_model.cuda()
    arcface_model.eval()


    ### process data
    folder_names = sorted(os.listdir(args.root_path))

    if args.start_id is not None:                              # resume process
        start_folder_name = 'n'+args.start_id.zfill(6)
        start_index = folder_names.index(start_folder_name)
        folder_names = folder_names[start_index:]
    
    pbar = tqdm(folder_names, 
                 total=len(folder_names),
                 leave=True, 
                 ncols=100, 
                 unit_scale=False, 
                 colour="white")

    for idx, folder_name in enumerate(pbar):
        pbar.set_description(f"No.{idx}")
        pbar.set_postfix({"正在处理": folder_name})

        folder_path = os.path.join(args.root_path, folder_name)
        save_folder_path = os.path.join(args.save_root_path, folder_name)

        preprocessAFolder(args, folder_path, save_folder_path, face_detector, emoca, face_parse_net, arcface_model)


if __name__ == '__main__':
    main()