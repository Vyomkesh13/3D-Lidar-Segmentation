import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

from src.nuscenes import NuScenes
from src.nuscenes.utils import splits

from src.pointcept.datasets.transform import Compose, TRANSFORMS

import mmcv
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose as mmsegCompose

from src.process_point_data import get_scene_samples_tokens, get_sample_points_labels
from src.process_image_data import get_sample_camera_images

from mmcv.parallel import collate


def get_point_data_by_lidar_token(nusc, lidar_data_token):
    # obtain the lidar sample data from nusc
    lidar_data = nusc.get('sample_data', lidar_data_token)
    # obtain sample from lidar data sample
    sample = nusc.get('sample', lidar_data['sample_token'])

    ################# 3D Point Data #################
    # obtain points and labels
    points, labels = get_sample_points_labels(nusc, sample)

    # coordinates of lidar point
    coord = points[:, :3]
    # strength of lidar point
    strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]
    # segmentation label
    segment = labels
    segment = np.vectorize(NuScenesDataset.get_learning_map(-1).__getitem__)(segment).astype(np.int64)

    # input data dictionary
    point_data_dict = dict(
        coord=coord, 
        strength=strength, 
        segment=segment,
        name=lidar_data_token)
    return point_data_dict

def get_point_data_by_sample_token(nusc, sample_token):
    # obtain sample from lidar data sample
    sample = nusc.get('sample', sample_token)

    ################# 3D Point Data #################
    # obtain points and labels
    points, labels = get_sample_points_labels(nusc, sample)

    # coordinates of lidar point
    coord = points[:, :3]
    # strength of lidar point
    strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]
    # segmentation label
    segment = labels
    segment = np.vectorize(NuScenesDataset.get_learning_map(-1).__getitem__)(segment).astype(np.int64)

    # input data dictionary
    point_data_dict = dict(
        coord=coord, 
        strength=strength, 
        segment=segment,
        name=sample_token)
    return point_data_dict

IMAGE_DATA_TEST_PIPELINE = [
    {
        "type": "MultiScaleFlipAug",
        "img_scale": (99999999, 640),
        "img_ratios": [1.0, 1.32, 1.73],
        "flip": True,
        "transforms": [
            {"type": "Resize", "keep_ratio": True},
            {"type": "RandomFlip"},
            {
                "type": "Normalize",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": True,
            },
            {"type": "ImageToTensor", "keys": ["img"]},
            {"type": "Collect", "keys": ["img"]},
        ],
    },
]

##################################### NuScenesDataset #####################################

class NuScenesDataset(Dataset):

    def __init__(
        self,
        split:str,
        data_root="./data/sets/nuscenes",
        transform=None,
        test_mode=False,
        test_cfg=None,
        ignore_index=-1,
        camera_image=False,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.learning_map = self.get_learning_map(ignore_index)
        self.camera_image = camera_image

        # test mode preparation
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        if camera_image:
            # build the data pipeline
            test_pipeline = [LoadImage()] + IMAGE_DATA_TEST_PIPELINE
            self.test_pipeline = mmsegCompose(test_pipeline)

        # create the NuScenes instance
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./data/sets/nuscenes', verbose=False)
        # obtain the samples coresponding to the split
        self.split_samples_tokens = self.get_split_samples_tokens()

    def get_split_scenes(self):
        """
        Obtain the scenes corresponding to the split
        """
        if self.split == 'train':
            split_scenes = splits.mini_train
        elif self.split == 'val':
            split_scenes = splits.mini_val[:1]
        elif self.split == 'test':
            split_scenes = splits.mini_val[1:]
        return split_scenes
    
    def get_split_samples_tokens(self):
        split_scenes = self.get_split_scenes()
        split_samples_tokens = []
        # iterate through each nusc scene
        for scene in self.nusc.scene:
            # obtain all samples tokens for the scene if it is in the specified split
            if scene['name'] in split_scenes:
                split_samples_tokens += get_scene_samples_tokens(self.nusc, scene)
        return split_samples_tokens
    
    def get_sample_token_by_idx(self, idx):
        # obtain the sample token from split
        sample_token = self.split_samples_tokens[idx]
        return sample_token
    
    def get_camera_data(self, camera_dict):
        """Reads image data from the cam dict provided.

        Args:
            camera_dict (Dict): Mapping from camera names to dict with image
                information ('data_path', 'sensor2lidar_translation',
                'sensor2lidar_rotation', 'cam_intrinsic').

                'data_path': data_path,
                'sample_data_token': sample_data_record['token'],
                'sensor2ego_translation': calibrated_sensor_record['translation'],
                'sensor2ego_rotation': calibrated_sensor_record['rotation'],
                'ego2global_translation': pose_record['translation'],
                'ego2global_rotation': pose_record['rotation'],
                'cam_intrinsic': camera intrinsic

        Returns:
            A dict with keys as camera names and the following attributes:
            'img': numpy array of image
            'lidar2cam_rt': lidar to camera transformation matrix
            'lidar2img_rt': lidar to image transformation matrix
            'cam_intrinsic': camera intrinsic
        """
        assert [Path(val['data_path']).exists() for _, val in camera_dict.items()]

        res_dict = dict()
        for cam in camera_dict.keys():
            res_dict[cam] = dict()
            res_dict[cam]['img'] = np.array(o3d.io.read_image(camera_dict[cam]['data_path']))
            
            # obtain lidar to cam transformation matrix
            lidar2cam_r = np.linalg.inv(camera_dict[cam]['sensor2lidar_rotation'])
            lidar2cam_t = camera_dict[cam]['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            intrinsic = camera_dict[cam]['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            # obtain lidar to image transformation matrix
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            res_dict[cam]['lidar2cam_rt'] = lidar2cam_rt
            res_dict[cam]['lidar2img_rt'] = lidar2img_rt
            res_dict[cam]['cam_intrinsic'] = camera_dict[cam]['cam_intrinsic']

        return res_dict

    def get_data(self, idx):
        """
        Get the data corresponding to the given index from the split dataset
        """
        # obtain the sample token from split
        sample_token = self.get_sample_token_by_idx(idx)
        # obtain the sample from nusc
        sample = self.nusc.get('sample', sample_token)

        ################# 3D Point Data #################
        # obtain points and labels
        points, labels = get_sample_points_labels(self.nusc, sample)

        # coordinates of lidar point
        coord = points[:, :3]
        # strength of lidar point
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]
        # segmentation label
        segment = labels.reshape([-1])
        segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int64)
        # lidar data token
        lidar_data_token = sample["data"]['LIDAR_TOP']
        
        # input data dictionary
        point_data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=lidar_data_token)

        ################# 2D Image Data #################
        # obtain camera image data
        camera_data_dict = {}
        if self.camera_image:
            camera_dict = get_sample_camera_images(self.nusc, sample)
            camera_data_dict = self.get_camera_data(camera_dict)

        return sample_token, point_data_dict, camera_data_dict
    
    # referenced inference_segmentor(model, imgs) from mmseg.apis.inference
    def transform_image_data(self, camera_data_dict):
        """Extract features from image(s) with DINOv2.

        Args:
            model (nn.Module): The loaded DINOv2 backbone.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            (list[Tensor]): list of transformed image data
        """
        result_camera_data_dict = deepcopy(camera_data_dict)
        if self.camera_image:            
            # transform the 6 camera images
            for cam in camera_data_dict.keys(): 
                # keep the original image
                result_camera_data_dict[cam]['ori_img'] = camera_data_dict[cam]['img']

                # transform the image
                img_data = dict(img=camera_data_dict[cam]['img'])
                img_data = self.test_pipeline(img_data)
                aug_img_data = collate([img_data], samples_per_gpu=1)

                # select the first out of 6 transformed images
                result_camera_data_dict[cam]['img'] = aug_img_data['img'][0]

                # select the second to last augmented image data
                result_camera_data_dict[cam]['img_metas'] = aug_img_data['img_metas'][0]

        return result_camera_data_dict

    def prepare_train_data(self, idx):
        # load training data
        sample_token, point_data_dict, camera_data_dict = self.get_data(idx)
        # transform point data
        point_data_dict = self.transform(point_data_dict)
        # transform image data
        camera_data_dict = self.transform_image_data(camera_data_dict)

        return sample_token, point_data_dict, camera_data_dict

    def prepare_test_data(self, idx):
        # load validation / test data
        sample_token, data_dict, camera_data_dict = self.get_data(idx)

        ################# Transform Point Data #################
        data_dict = self.transform(data_dict)

        ### test time transform with fragments of points ###
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list

        ################# Transform Image Data #################
        camera_data_dict = self.transform_image_data(camera_data_dict)
        
        return sample_token, result_dict, camera_data_dict

    @staticmethod
    def get_learning_map(ignore_index):
        """
        The nuScenes-lidarseg dataset comes with annotations for 32 classes (details). 
        Some of these only have a handful of samples. Hence we merge similar classes and remove rare classes. 
        This results in 16 classes for the lidar segmentation challenge.
        """
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map

    @staticmethod
    def get_reversed_learning_map(ignore_index):
        reversed_learning_map = {
            ignore_index: (0, 1, 5, 7, 8, 10, 11, 13, 19, 20, 29, 31),
            6: (2, 3, 4, 6),
            0: (9,),
            7: (12,),
            1: (14,),
            2: (15, 16),
            3: (17,),
            4: (18,),
            5: (21,),
            8: (22,),
            9: (23,),
            10: (24,),
            11: (25,),
            12: (26,),
            13: (27,),
            14: (28,),
            15: (30,)
        }
        return reversed_learning_map
    
    @staticmethod
    def get_reversed_learning_map_single(ignore_index):
        reversed_learning_map = {
            ignore_index: 0,
            6: 2,
            0: 9,
            7: 12,
            1: 14,
            2: 15,
            3: 17,
            4: 18,
            5: 21,
            8: 22,
            9: 23,
            10: 24,
            11: 25,
            12: 26,
            13: 27,
            14: 28,
            15: 30
        }
        return reversed_learning_map

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.split_samples_tokens)
    