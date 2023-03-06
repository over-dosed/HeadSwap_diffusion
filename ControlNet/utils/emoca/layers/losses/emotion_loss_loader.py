import copy

import omegaconf
import torch
from HSD.utils.emoca.layers.losses.EmonetLoader import get_emonet
from pathlib import Path
import torch.nn.functional as F
try:
    from HSD.utils.emoca.models.EmoNetModule import EmoNetModule
except ImportError as e:
    print(f"Could not import EmoNetModule. EmoNet models will not be available. Make sure you pull the repository with submodules to enable EmoNet.")
try:
    from HSD.utils.emoca.models.EmoSwinModule import EmoSwinModule
except ImportError as e: 
    print(f"Could not import EmoSwinModule. SWIN models will not be available. Make sure you pull the repository with submodules to enable SWIN.")
from HSD.utils.emoca.models.EmoCnnModule import EmoCnnModule
from HSD.utils.emoca.models.EmoDECA import EmoDECA
from HSD.utils.emoca.models.IO import get_checkpoint_with_kwargs
from HSD.utils.emoca.utils.other import class_from_str
import sys


def emo_network_from_path(path):
    print(f"Loading trained emotion network from: '{path}'")

    def load_configs(run_path):
        from omegaconf import OmegaConf
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        if run_path != conf.inout.full_run_dir: 
            conf.inout.output_dir = str(Path(run_path).parent)
            conf.inout.full_run_dir = str(run_path)
            conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
        return conf

    cfg = load_configs(path)

    if not bool(cfg.inout.checkpoint_dir):
        cfg.inout.checkpoint_dir = str(Path(path) / "checkpoints")

    checkpoint_mode = 'best'
    stages_prefixes = ""

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes,
                                                               checkpoint_mode=checkpoint_mode,
                                                               # relative_to=relative_to_path,
                                                               # replace_root=replace_root_path
                                                               )
    checkpoint_kwargs = checkpoint_kwargs or {}

    if 'emodeca_type' in cfg.model.keys():
        module_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        module_class = EmoNetModule

    emonet_module = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False,
                                                      **checkpoint_kwargs)
    return emonet_module
