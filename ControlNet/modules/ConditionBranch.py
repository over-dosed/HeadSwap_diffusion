import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.DECA.decalib.deca import DECA
from utils.DECA.decalib.utils.config import cfg as deca_cfg


class Condition_Branch(nn.Module):

    def __init__(self, device = 'cuda:0'):
        super().__init__() 

        # DECA for rendering condition images
        deca_cfg.model.use_tex = True             
        deca_cfg.rasterizer_type = 'pytorch3d'  #  help='rasterizer type: pytorch3d or standard' 
        deca_cfg.model.extract_tex = True
        self.deca = DECA(config = deca_cfg, device= device)
        self.deca.eval()
        

    def forward(self, codedict):
        # codedict_source & codedict_target: python dict of {'tforms', 'shape', 'tex', 'exp', 'pose', 'cam', 'light'}, tensor, cuda
        # each item has shape (B, _)
        # original_images: tensor, cuda, B*3*512*512, RGB, 0~1

        with torch.no_grad():
            for key in codedict:
                original_device = codedict[key].device
                codedict[key] = codedict[key].to(self.deca.device)
            render_image = self.deca.render_for_hsd(codedict)  # (B, 3, 512, 512), tensor, GPU, 0~1
            # render_image = Image.fromarray((reder_image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))
            render_image = render_image.to(original_device)

        return render_image