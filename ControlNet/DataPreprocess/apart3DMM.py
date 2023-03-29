## this is a simple script for apart 3DMM pkl file

# original 3DMM pkl:
# tforms, shape, tex, exp, pose, cam, light, verts, trans_verts (512 size), landmarks2d (512 size), landmarks3d (512 size)

# apart 
# 1. tforms, shape, tex, exp, pose, cam, light           [for condition branch]
# 2. trans_verts                                         [for feature branch]

import os
import pickle
from tqdm import tqdm

condition_keys = ['tforms', 'shape', 'tex', 'exp', 'pose', 'cam', 'light']
feature_keys = ['trans_verts']

def process3DMM(folder_path):
    file_path = os.path.join(folder_path, '3DMM.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f_read:
            data = pickle.load(f_read)

        condition_data = {key:data[key].cpu() for key in condition_keys}
        feature_data = {key:data[key].cpu() for key in feature_keys}

        with open(os.path.join(folder_path, '3DMM_condition.pkl'), 'wb') as f_write_1:
            pickle.dump(condition_data, f_write_1)

        with open(os.path.join(folder_path, '3DMM_feature.pkl'), 'wb') as f_write_2:
            pickle.dump(feature_data, f_write_2)

    else:
        print('not find 3dmm in {}'.format(folder_path))
        exit(-1)

    return

def apart3DMM(root_path):

    folder_names = sorted(os.listdir(root_path))
    pbar = tqdm(folder_names, 
                 total=len(folder_names),
                 leave=True, 
                 ncols=100, 
                 unit_scale=False, 
                 colour="white")

    for idx, folder_name in enumerate(pbar):
        pbar.set_description(f"No.{idx}")
        pbar.set_postfix({"正在处理": folder_name})

        folder_path = os.path.join(root_path, folder_name)

        process3DMM(folder_path)


if __name__ == '__main__':
    apart3DMM('/data0/wc_data/VFHQ/train')