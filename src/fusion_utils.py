import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.extract_features import CAMERA_TYPES
from src.model import offset2batch, batch2offset

def pad_batch_voxels(point_feature_dict, feat_key='feat'):
    # batch size obtained from the number of offsets
    batch_size = point_feature_dict['offset'].shape[0]

    # point feat output from ptv3 (batch_num_points, 64)
    point_features = point_feature_dict[feat_key]

    num_batch_voxels, point_feat_dim = point_features.shape

    # offset to batch
    batch_points = offset2batch(point_feature_dict['offset'])
    # maximum number of voxels out of those in batch
    max_num_voxels = int(batch_points.bincount().max().item())

    # padded point features
    padded_point_features = torch.full((batch_size, max_num_voxels, point_feat_dim), 
                                        fill_value=0.0,
                                        dtype=point_features.dtype,
                                        device=point_features.device)
    # padding mask: 0 padded, 1 unpadded
    padding_mask = torch.zeros((batch_size, max_num_voxels), dtype=torch.bool)
    # list of point features for each sample in batch
    batch_point_features_list = list(torch.tensor_split(point_features, point_feature_dict['offset']))[:-1]
    # obtain padded point feature of shape (batch_size, max_num_voxels, 64)
    for i, point_feat in enumerate(batch_point_features_list):
        num_real_vox = min(point_feat.shape[0], max_num_voxels)
        padded_point_features[i, :num_real_vox] = point_feat[:num_real_vox]
        padding_mask[i, :num_real_vox] = True
    
    return padded_point_features, padding_mask

def project_point_to_pixel(sample_token, point_feature_dict, image_feature_dict):
    """
    Project points in point_feature_dict to pixels in each of the 6 images in image_feature_dict
    Batch operations
    Used by DINOPT model to fuse and learn both point and image features
    
    Args:
        sample_token: sample token
        point_feature_dict: dictionary containing the extracted point features
        image_feature_dict: dictionary containing the extracted image features
    Returns:
        projected pixels from the points for all samples in batch
    """
    # dictionary storing the results of point to pixel projection for each camera
    cam_to_projections = {}

    # iterate through each camera
    for camera in CAMERA_TYPES:

        # coordinates of voxels (batch_size, max_num_voxels, 3)
        padded_point_coord, padding_mask = pad_batch_voxels(point_feature_dict, 'coord')
        batch_size, max_num_voxels, point_feat_dim = padded_point_coord.shape
        
        # original camera image (batch_size, 900, 1600, 3)
        ori_img = image_feature_dict[camera]['ori_img']
        batch_size, H, W, C = ori_img.shape

        # transformed camera image (batch_size, 3, resized_H, resized_W)
        img = image_feature_dict[camera]['img']
        batch_size, C, resized_H, resized_W = img.shape

        # lidar to camera transformation matrix
        lidar2cam_rt = image_feature_dict[camera]['lidar2cam_rt'].float()

        # lidar to image transformation matrix
        lidar2img_rt = image_feature_dict[camera]['lidar2img_rt'].float()

        # points in homogeneous coordinates
        homogeneous_ones = torch.ones((batch_size, max_num_voxels, 1))
        points_coords_homogeneous = torch.cat((padded_point_coord, homogeneous_ones), dim=-1)
       
        # project point to image pixel coordinate
        projected_uvw = torch.bmm(points_coords_homogeneous, lidar2img_rt.permute((0, 2, 1)))
        projected_uvw = projected_uvw[:, :, :3]
        # normalize
        projected_uvw = projected_uvw / projected_uvw[:, :, 2:3].repeat(1, 1, 3).reshape(batch_size, max_num_voxels, 3)
        projected_uv = projected_uvw[:, :, :2]

        # depths: project point to camera coordinate
        depths = torch.bmm(points_coords_homogeneous, lidar2cam_rt)
        depths = depths[:, :, 2]

        # projection mask: 1 if projected pixel is in image boundaries, 0 otherwise
        projection_mask = torch.ones((depths.shape[0], depths.shape[1]), dtype=torch.bool)
        projection_mask = torch.logical_and(projection_mask, depths > 1)
        projection_mask = torch.logical_and(projection_mask, projected_uv[:, :, 0] > 1)
        projection_mask = torch.logical_and(projection_mask, projected_uv[:, :, 0] < W - 1)
        projection_mask = torch.logical_and(projection_mask, projected_uv[:, :, 1] > 1)
        projection_mask = torch.logical_and(projection_mask, projected_uv[:, :, 1] < H - 1)
        # set the position to False if it is a padded position
        assert padding_mask.shape == projection_mask.shape, f"Mismatch in padding mask shape and projection mask shape"
        projection_mask = torch.where(padding_mask == 1, projection_mask, False)
        projected_uv = projected_uv[projection_mask]

        cam_to_projections[camera] = (projected_uv, projection_mask)

    return cam_to_projections

def get_projection_offset(projection_mask):
    num_projected_per_sample = projection_mask.sum(dim=-1)
    projection_offset = torch.cumsum(num_projected_per_sample, dim=0)
    return projection_offset

def get_pixel_patch(cam_to_projections, image_feature_dict, patch_size=14):
    """
    Get the position of the image patch containing pixel u, v

    Args:
        cam_to_projections: dictionary mapping camera to (projected_uv, projection_mask)
            projected_uv: projected pixel from point, shape (num_batch_pixels, 2)
                u: column index of the pixel in the original camera image
                v: row index of the pixel in the original camera image
            projection_mask: projection mask, shape (batch_size, max_num_voxels)
                1 if projected pixel is in image boundaries
                0 if projected pixel outside image boundaries
        image_feature_dict: dictionary containing the extracted image features
        patch_size: height and width of the square image patch
    Returns:
        dictionary mapping camera type to the position index of the image patch containing the projected pixel
    """
    cam_to_projection_patch = {}

    # iterate through each camera
    for camera in CAMERA_TYPES:
        projected_uv, projection_mask = cam_to_projections[camera]
        # offset indicating the start of projected pixels of the sample
        projection_offset = get_projection_offset(projection_mask)

        # original camera image (batch_size, 900, 1600, 3)
        ori_img = image_feature_dict[camera]['ori_img']
        batch_size, H, W, C = ori_img.shape

        # transformed camera image (batch_size, 3, resized_H, resized_W)
        img = image_feature_dict[camera]['img']
        batch_size, C, resized_H, resized_W = img.shape

        # extracted camera image patch token features (batch_size, img_feat_dim, num_height_patch, num_width_path)
        patch_token_feat = image_feature_dict[camera]['patch_token_feat']
        batch_size, img_feat_dim, num_height_patch, num_width_path = patch_token_feat.shape

        # obtain the pixel position in the resized image resolution
        rescaled_projected_uv = torch.zeros((projected_uv.shape[0], projected_uv.shape[1]), device=projected_uv.device)
        rescaled_projected_uv[:, 0] = torch.floor(projected_uv[:, 0] / W * resized_W)
        rescaled_projected_uv[:, 1] = torch.floor(projected_uv[:, 1] / H * resized_H)

        # convert pixel coordinate (u, v) to patch index (i, j)
        patch_ij = torch.floor(rescaled_projected_uv / patch_size).to(torch.int32)

        assert (patch_ij[:, 0] >= 0).all() and (patch_ij[:, 0] < num_width_path).all()
        assert (patch_ij[:, 1] >= 0).all() and (patch_ij[:, 1] < num_height_patch).all()

        # update dictionary with patch_ij, shape (num_batch_projected_pixels, 2)
        cam_to_projection_patch[camera] = patch_ij

    return cam_to_projection_patch

def extract_associated_point_image_features(sample_token, point_feature_dict, image_feature_dict, cam_to_projections, cam_to_projection_patch):
    """
    Extract the point-associated image feature

    Args:
        sample_token: sample token
        point_feature_dict: dictionary containing the extracted point features
        image_feature_dict: dictionary containing the extracted image features
        cam_to_projections: dictionary mapping camera to (projected_uv, projection_mask)
        cam_to_projection_patch: dictionary mapping camera type to the position index of the image patch containing the projected pixel
    Returns:
        dictionary mapping camera type to the extracted image patch feature associated to the voxels projectable to the camera image
            for each camera, the associated_image_feature has shape (batch_size, max_num_voxels, img_feat_dim)
    """
    cam_to_associated_image_feature = {}
    for camera in CAMERA_TYPES:
        # coordinates of voxels (batch_size, max_num_voxels, 3)
        padded_point_coord, padding_mask = pad_batch_voxels(point_feature_dict, 'coord')
        batch_size, max_num_voxels, point_feat_dim = padded_point_coord.shape

        # extracted camera image patch token features (batch_size, img_feat_dim, num_height_patch, num_width_path)
        patch_token_feat = image_feature_dict[camera]['patch_token_feat']
        batch_size, img_feat_dim, num_height_patch, num_width_path = patch_token_feat.shape

        # projected pixel in original image and projection mask
        projected_uv, projection_mask = cam_to_projections[camera]

        # offset indicating the start of projected pixels of the sample
        projection_offset = get_projection_offset(projection_mask)

        # shape (num_batch_projected_pixels, 2)
        patch_ij = cam_to_projection_patch[camera]
        # list of patch_ij for each of the sample in batch
        patch_ij_list = list(torch.tensor_split(patch_ij, projection_offset))[:-1]

        # batch index corresponding to each patch index in patch_ij: [0, ..., 0, 1, ..., 1, ..., B, ..., B]
        b_indices = [torch.full((p_ij.shape[0], 1), batch_idx) for batch_idx, p_ij in enumerate(patch_ij_list)]
        b_indices = torch.cat(b_indices, dim=0)

        # (batch_size, max_num_voxels, img_feat_dim)
        associated_image_feature = torch.zeros((batch_size, max_num_voxels, img_feat_dim))

        # extracted image patch feature for projected voxels
        projection_associated_patch_feat = patch_token_feat.permute(0, 3, 2, 1)[b_indices[:, 0], patch_ij[:, 0], patch_ij[:, 1]]

        assert projection_mask.sum() == projection_associated_patch_feat.shape[0]

        # set the image patch feature of projected voxels
        associated_image_feature[projection_mask] = projection_associated_patch_feat

        # update result dictionary
        cam_to_associated_image_feature[camera] = associated_image_feature
    
    return cam_to_associated_image_feature