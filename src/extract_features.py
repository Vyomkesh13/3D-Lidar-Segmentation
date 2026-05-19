import torch
import numpy as np
from PIL import Image
import spconv.pytorch as spconv
from functools import partial

from src.load_dinov2_model import CenterPadding
from mmcv.parallel import scatter

CAMERA_TYPES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    ]

point_feat_extract_keys = ['coord', 'grid_coord', 'segment', 'offset', 'feat']

#########################################################
#                   batch features                      #
#########################################################
def extract_batch_point_features(ptv3_model,
                                 point_data_dict:dict):
    # move point data to cuda
    for key, item in point_data_dict.items():
        if isinstance(item, torch.Tensor):
            point_data_dict[key] = point_data_dict[key].cuda(non_blocking=True)

    ################## run ptv3 model on point data ##################
    with torch.no_grad():
        point_feature_dict = ptv3_model(point_data_dict)
    
    # move point feature output from cuda to cpu
    for key, item in point_feature_dict.items():
        if isinstance(item, torch.Tensor):
            point_feature_dict[key] = point_feature_dict[key].cpu()
        elif isinstance(item, spconv.SparseConvTensor):
            point_feature_dict[key] = point_feature_dict[key].replace_feature(point_feature_dict[key].features.cpu())
            point_feature_dict[key].indices = point_feature_dict[key].indices.cpu()
    
    # move point data back to cpu
    for key, item in point_data_dict.items():
        if isinstance(item, torch.Tensor):
            point_data_dict[key] = point_data_dict[key].cpu()

    ################## extract the needed point features for joint modeling ##################
    # dictionary mapping field name to value of point features to be used for joint modeling
    joint_point_feature_dict = {}
    for ext_key in point_feat_extract_keys:
        joint_point_feature_dict[ext_key] = point_feature_dict[ext_key]
    # include the inverse and origin_segment from the point input data
    joint_point_feature_dict['inverse'] = point_data_dict['inverse']
    joint_point_feature_dict['origin_segment'] = point_data_dict['origin_segment']

    return joint_point_feature_dict


def get_feature_extract_model(dinov2_backbone_model, dinov2_config):
    class ImgFeatExtractModel:
        def __init__(self, backbone):
            self.backbone = backbone

    img_feat_extract_model = ImgFeatExtractModel(dinov2_backbone_model)
            
    # define the forward inference with only the last 4 
    img_feat_extract_model.backbone.forward = partial(
        dinov2_backbone_model.get_intermediate_layers,
        n=dinov2_config.model.backbone.out_indices, 
        reshape=True,
    )
    if hasattr(dinov2_backbone_model, "patch_size"):
        img_feat_extract_model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(dinov2_backbone_model.patch_size)(x[0]))

    return img_feat_extract_model

def extract_batch_image_features(dinov2_model,
                                 camera_data_dict:dict):
    joint_image_feature_dict = {}
    with torch.no_grad():
        # iterate through each camera image   
        for cam in camera_data_dict.keys():
            # move the transformed image to cuda
            camera_data_dict[cam]['img'] = camera_data_dict[cam]['img'].cuda(non_blocking=True)
            
            ################## run dinov2 model on image data ##################
            patch_tokens = dinov2_model.backbone(camera_data_dict[cam]['img'])
            patch_tokens = torch.concat(patch_tokens, dim=1)

            # move patch tokens to cpu
            patch_tokens = patch_tokens.cpu()

            # move the transformed image back to cpu
            camera_data_dict[cam]['img'] = camera_data_dict[cam]['img'].cpu()

            # update image feature output dictionary
            joint_image_feature_dict[cam] = {}
            joint_image_feature_dict[cam]['patch_token_feat'] = patch_tokens
            # extract the original image, transformed image, image metadata, rotation transformation matrices
            for ext_key in camera_data_dict[cam].keys():
                joint_image_feature_dict[cam][ext_key] = camera_data_dict[cam][ext_key]

    return joint_image_feature_dict

#########################################################
#               per-sample features                     #
#########################################################
def process_point_features_for_joint_modeling(ptv3_model, 
                                              sample_token:str, 
                                              point_data_dict:dict, 
                                              sample_tok_to_point_feature_outputs:dict):    
    # move point data to cuda
    for key, item in point_data_dict.items():
        if isinstance(item, torch.Tensor):
            point_data_dict[key] = point_data_dict[key].cuda(non_blocking=True)
    
    ################## run ptv3 model on point data ##################
    with torch.no_grad():
        point_feature_dict = ptv3_model(point_data_dict)

    # move point feature output from cuda to cpu
    for key, item in point_feature_dict.items():
        if isinstance(item, torch.Tensor):
            point_feature_dict[key] = point_feature_dict[key].cpu()
        elif isinstance(item, spconv.SparseConvTensor):
            point_feature_dict[key] = point_feature_dict[key].replace_feature(point_feature_dict[key].features.cpu())
            point_feature_dict[key].indices = point_feature_dict[key].indices.cpu()

    # move point data back to cpu
    for key, item in point_data_dict.items():
        if isinstance(item, torch.Tensor):
            point_data_dict[key] = point_data_dict[key].cpu()

    ################## extract the needed model output ##################
    extract_keys = ['coord', 'grid_coord', 'segment', 'feat']

    curr_sample_offset = 0
    for i, sample_tok in enumerate(sample_token):
        print("sample {} - offset {}".format(i, curr_sample_offset))
        
        # initialize an empty dictionary for the sample in current iteration to store selected point output
        single_sample_point_feature_dict = {}

        curr_sample_num_points = point_feature_dict['batch'].bincount()[i]
        assert curr_sample_num_points == point_feature_dict['offset'][i] - curr_sample_offset

        # break down the point feature dict into one dict per sample
        for point_output_key in extract_keys:
            # extract the point output for the current sample
            point_output_item = point_feature_dict[point_output_key][curr_sample_offset : point_feature_dict['offset'][i]]
            # update the current sample's dictionary
            single_sample_point_feature_dict[point_output_key] = point_output_item

        # keep the inverse used to convert voxelized points back to original points
        single_sample_point_feature_dict['inverse'] = point_data_dict['inverse'][curr_sample_offset : point_feature_dict['offset'][i]]

        # keep the origin_segment used to compare the predicted segement with
        single_sample_point_feature_dict['origin_segment'] = point_data_dict['inverse'][curr_sample_offset : point_feature_dict['offset'][i]]

        # update the dictionary to be used for combined feature learning
        sample_tok_to_point_feature_outputs[sample_tok] = single_sample_point_feature_dict

        curr_sample_offset = point_feature_dict['offset'][i]

        assert set(single_sample_point_feature_dict.keys()) == set(('coord', 'grid_coord', 'segment', 'feat', 'inverse', 'origin_segment'))


import mmcv
from mmseg.apis.inference import LoadImage
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

# referenced inference_segmentor(model, imgs) from mmseg.apis.inference
def prep_img_data(model, cfg, imgs):
    """Extract features from image(s) with DINOv2.

    Args:
        model (nn.Module): The loaded DINOv2 backbone.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    if isinstance(cfg, str):
        cfg = mmcv.Config.fromfile(cfg)
    elif not isinstance(cfg, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(cfg)))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)

    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    return data

def process_image_features_for_joint_modeling(dinov2_backbone_model, 
                                              dinov2_cfg,
                                              sample_token:str,
                                              camera_data_dict:dict, 
                                              sample_tok_to_image_feature_output:dict):
        
    # iterate through each camera 
    for camera_name, camera_data_dict_list in camera_data_dict.items():

        # iterate through the data for each sample
        for i, (sample_tok, single_camera_data_dict) in enumerate(zip(sample_token, camera_data_dict_list)):
            print("sample {}".format(i))

            # creates an image from array in RGB
            image = Image.fromarray(single_camera_data_dict['img']).convert("RGB")
            # image array in BGR
            img_array = np.array(image)[:, :, ::-1] # BGR
            # preprocess image data according to test pipeline defined in dinov2 config (TODO:may need to redesign the preprocess)
            img_data = prep_img_data(dinov2_backbone_model, dinov2_cfg, img_array)

            # choose one augmented image out of the 6 augmented image
            img_tensor = img_data['img'][-2]
            img_tensor = img_tensor.cuda(non_blocking=True)
            
            class Model:
                def __init__(self, backbone) -> None:
                    self.backbone = backbone

            model = Model(dinov2_backbone_model)

            with torch.no_grad():
                # define the forward inference with only the last 4 
                model.backbone.forward = partial(
                    dinov2_backbone_model.get_intermediate_layers,
                    n=dinov2_cfg.model.backbone.out_indices, 
                    reshape=True,
                )
                if hasattr(dinov2_backbone_model, "patch_size"):
                    model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(dinov2_backbone_model.patch_size)(x[0]))

                all_tokens = model.backbone(img_tensor)[0]

                img_tensor = img_tensor.cpu()
                all_tokens = all_tokens.cpu()

                sample_tok_to_image_feature_output[sample_tok][camera_name]['patch_token_feat'] = all_tokens