import os
import torch
import multiprocessing as mp

from src.model import PointTransformerV3
from src.segmentor import DefaultSegmentorV2
# from src.pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
# from src.pointcept.models.default import DefaultSegmentorV2
from collections import OrderedDict
from huggingface_hub import hf_hub_download
from src.pointcept.engines.defaults import default_config_parser
import src.pointcept.utils.comm as comm
from src.pointcept.utils.env import set_seed

def default_ptv3_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed)
    return cfg

def load_ptv3_ckpt_and_config(
        name: str = "nuscenes-semseg-pt-v3m1-0-base/model/model_best",
        config_name: str = "nuscenes-semseg-pt-v3m1-0-base/config",
        repo_id="Pointcept/PointTransformerV3",
        download_root: str = "exp/nuscenes",
        custom_config: dict = None,
        enable_test_time_aug: bool = False
    ):
    print(f"Loading checkpoint from HuggingFace: {name} ...")
    # download checkpoint file
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{name}.pth",
        repo_type="model",
        revision="main",
        local_dir=download_root or os.path.expanduser("~/.cache/pointtransformerv3/ckpt"),
    )
    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"Loading config from HuggingFace: {name} ...")
    # download config file
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{config_name}.py",
        repo_type="model",
        revision="main",
        local_dir=download_root or os.path.expanduser("~/.cache/pointtransformerv3/ckpt"),
    )
    # parse and obtain config
    options = dict(
        save_path="exp/nuscenes/nuscenes-semseg-pt-v3m1-0-base",
        weight=ckpt_path
        )
    config = default_config_parser(config_path, options)
    config = default_ptv3_setup(config)
    
    # update config to include custom config
    if custom_config is not None:
        for key, value in custom_config.items():
            if key == "enable_flash":
                config['model']['backbone']['enable_flash'] = value
            else:
                config[key] = value

    # disable test time data augmentation
    if not enable_test_time_aug:
        config.data.test.test_cfg.aug_transform = [
                    [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
                ]

    return ckpt, config

def get_ptv3_backbone_weight(ckpt):
    weight = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        if key.startswith("module.backbone."):
            if comm.get_world_size() == 1:
                key = key[16:]  # module.backbone.xxx.xxx -> xxx.xxx
        elif key.startswith("module."):
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
        else:
            if comm.get_world_size() > 1:
                key = "module." + key  # xxx.xxx -> module.xxx.xxx

        if key in ('seg_head.weight', 'seg_head.bias'):
            continue
        weight[key] = value
    return weight
    
def load_ptv3_backbone(ckpt, config):
    # model config
    model_config = config['model']['backbone'].deepcopy()
    model_config.pop('type')
    # initialize model with loaded config
    model = PointTransformerV3(**model_config)

    # model weights
    weight = get_ptv3_backbone_weight(ckpt)

    # load model with weights
    model.load_state_dict(weight)
    # number of model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model

def load_ptv3_segmentor(ckpt, config):
    # model config
    model_config = config['model'].deepcopy()
    model_config.pop('type')
    # initialize model with loaded config
    model = DefaultSegmentorV2(**model_config)

    # model weights
    weight = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        if key.startswith("module."):
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
        else:
            if comm.get_world_size() > 1:
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
        weight[key] = value

    # load model with weights
    model.load_state_dict(weight)
    # number of model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model