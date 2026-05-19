import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from pathlib import Path

from src.dataset_nuscenes import NuScenesDataset

from src.extract_features import extract_batch_point_features, extract_batch_image_features

def extract_and_save_features(ptv3_feat_extract_model, 
                              dinov2_feat_extract_model, 
                              dataloader:DataLoader,
                              feat_root_path='feat/nuscenes'):
    if not os.path.exists(feat_root_path):
        os.makedirs(feat_root_path, exist_ok=True)

    # iterate through each training sample
    for sample_token, point_data_dict, camera_data_dict in tqdm(dataloader):
        sample_token = sample_token[0]
        
        # file to save the sample features
        sample_feature_filepath = os.path.join(feat_root_path, f"{sample_token}.pt")
        # extract point features of the current training sample
        point_feature_dict = extract_batch_point_features(ptv3_feat_extract_model, point_data_dict)
        # extract image features of the current training sample
        image_feature_dict = extract_batch_image_features(dinov2_feat_extract_model, camera_data_dict)

        save_dict = dict(sample_token=sample_token, 
                         point_feature_dict=point_feature_dict,
                         image_feature_dict=image_feature_dict)
        
        # save to filepath
        torch.save(save_dict, sample_feature_filepath)

##################################### NuScenesJointFeature #####################################

class NuScenesFeatureDataset(Dataset):

    def __init__(
        self,
        nuscenes_dataset:NuScenesDataset,
        feat_root_path='feat/nuscenes',
        patch_feat_layers=[0],
        img_feat_dim=384
    ):
        super().__init__()
        self.nuscenes_dataset = nuscenes_dataset
        self.feat_root_path = feat_root_path

        self.patch_feat_layers = patch_feat_layers
        patch_feat_indices = [layer_idx * img_feat_dim + np.arange(img_feat_dim) for layer_idx in patch_feat_layers]
        patch_feat_indices = np.hstack(patch_feat_indices)
        self.patch_feat_indices = torch.from_numpy(patch_feat_indices)

    def get_feature(self, idx):
        # get the sample token at given idx
        sample_token = self.nuscenes_dataset.get_sample_token_by_idx(idx)

        sample_feature_path = os.path.join(self.feat_root_path, f"{sample_token}.pt")
        
        assert os.path.exists(sample_feature_path), f"Feature filepath does not exist: {sample_feature_path}"

        # load the saved feature
        save_dict = torch.load(sample_feature_path, weights_only=False)

        assert save_dict['sample_token'] == sample_token, f"Mismatch between sample token at idx and sample token saved in {sample_feature_path}"

        # obtain the point and image feature dictionaries
        point_feature_dict = save_dict['point_feature_dict']
        image_feature_dict = save_dict['image_feature_dict']

        point_feature_dict['origin_offset'] = torch.tensor([point_feature_dict['origin_segment'].shape[0]])

        # image patch features obtained form 4 layers - get specified layer's feature
        for camera in image_feature_dict.keys():
            image_feature_dict[camera]['patch_token_feat'] = image_feature_dict[camera]['patch_token_feat'][:, self.patch_feat_indices]

        return sample_token, point_feature_dict, image_feature_dict
        

    def __getitem__(self, idx):
        return self.get_feature(idx)

    def __len__(self):
        return len(self.nuscenes_dataset.split_samples_tokens)
    