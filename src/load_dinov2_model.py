import os
import math
import torch
import torch.nn.functional as F
import itertools
import urllib.request
import mmcv


class CenterPadding(torch.nn.Module):
    """
    CenterPadding applied to ensure image can be partitioned into patch size
    """
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
    
def load_dinov2_backbone(backbone_size="small", with_registers=False):
    """
    Load DINOv2 Backbone based on provided specifications

    Args:
        backbone_size: small (21 M params), base (86 M params), large (300 M params), giant (1100 M params)
        with_registers: whether the model with registers or not
    """
    dinov2_backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    dinov2_backbone_arch = dinov2_backbone_archs[backbone_size]
    dinov2_backbone_reg = "_reg" if with_registers else ""
    dinov2_backbone_name = f"dinov2_{dinov2_backbone_arch}{dinov2_backbone_reg}"

    # load pretrained backbone model
    dinov2_backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=dinov2_backbone_name)
    # number of model parameters
    n_parameters = sum(p.numel() for p in dinov2_backbone_model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return dinov2_backbone_model, dinov2_backbone_name

def load_dinov2_config(dinov2_backbone_name:str, 
                       head_scale_count=3, 
                       head_dataset="ade20k", 
                       head_type="ms"):
    """
    Load DINOv2 Config based on specified head dataset

    Args:
        dinov2_backbone_name: name of DINOv2 backbone (obtained from calling load_dinov2_backbone)
        head_scale_count: more scales: slower but better results, in (1,2,3,4,5)
        head_dataset: in ("ade20k", "voc2012")
        head_type: in ("ms, "linear")
    """
    def load_config_from_url(url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
        
    assert head_dataset in ("ade20k", "voc2012"), f"Unexpected head dataset: {head_dataset}"
    assert head_type in ("ms", "linear"), f"Unexpected head type: {head_type}"

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    dinov2_head_config_url = f"{DINOV2_BASE_URL}/{dinov2_backbone_name}/{dinov2_backbone_name}_{head_dataset}_{head_type}_config.py"

    dinov2_cfg_str = load_config_from_url(dinov2_head_config_url)
    dinov2_cfg = mmcv.Config.fromstring(dinov2_cfg_str, file_format=".py")
    if head_type == "ms":
        dinov2_cfg.data.test.pipeline[1]["img_ratios"] = dinov2_cfg.data.test.pipeline[1]["img_ratios"][:head_scale_count]
        print("scales:", dinov2_cfg.data.test.pipeline[1]["img_ratios"])
    
    return dinov2_cfg