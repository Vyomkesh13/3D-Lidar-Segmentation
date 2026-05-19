import os
from typing import List, Dict, Any, Tuple

from src.nuscenes import NuScenes
from src.nuscenes.utils import splits
from src.nuscenes.utils.data_classes import LidarSegPointCloud
from src.pointcept.datasets.transform import Compose, TRANSFORMS

def get_scene_samples_tokens(nusc:NuScenes, scene:Dict[str, Any], verbose=False) -> List[str]:
    """
    Get tokens of all the samples in scene
    """
    if verbose:
        print("Scene: \n{}".format(scene))

    # first sample of scene
    sample = nusc.get('sample', scene['first_sample_token'])
    # a list of all samples of the scene
    scene_samples_tokens = [sample['token']]
    while sample['next'] != '':
        sample =  nusc.get('sample', sample['next'])
        scene_samples_tokens += [sample['token']]

    if verbose:
        print("Scene {} has {} samples".format(scene['token'], len(scene_samples_tokens)))
        print("Tokens of all samples of scene: \n{}".format(scene_samples_tokens))

    # check if the last sample in scene has been obtained correctly
    assert scene['last_sample_token'] == scene_samples_tokens[-1], \
        f'Token of the last sample obtained from iteration through all samples in scene is incorrect'

    return scene_samples_tokens

def get_sample_points_labels(nusc:NuScenes, sample:Dict[str, Any], gt_from='lidarseg', verbose=False) -> Tuple:
    """
    Get all the points and labels of the sample
    """
    sample_token = sample['token']
    if verbose:
        print("Sample token: {}".format(sample_token))
    
    ################# Lidar Sensor #################
    # pointsensor channel
    pointsensor_channel = 'LIDAR_TOP'
    # token of point sensor
    pointsensor_token = sample['data'][pointsensor_channel]
    # sample data of point sensor
    pointsensor = nusc.get('sample_data', pointsensor_token)

    ################ Filepath to Points and Labels ################
    points_filepath = os.path.join(nusc.dataroot, pointsensor['filename'])
    lidarseg_labels_filepath = os.path.join(nusc.dataroot, nusc.get(gt_from, pointsensor_token)['filename'])

    ################ Obtains Points and Labels ################
    if pointsensor['sensor_modality'] == 'lidar':
        assert hasattr(nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'
        # Ensure that lidar pointcloud is from a keyframe
        assert pointsensor['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
        # create the LidarSegPointCloud to load points and labels
        lidar_seg_point_cloud = LidarSegPointCloud(points_filepath, lidarseg_labels_filepath)

        if verbose:
            if lidar_seg_point_cloud.points == None:
                print("Point cloud points loading failed, got {}".format(lidar_seg_point_cloud.points))
            else:
                print("Point cloud points shape: {}".format(lidar_seg_point_cloud.points.shape))
                print("Point cloud points: \n{}".format(lidar_seg_point_cloud.points))
            if lidar_seg_point_cloud.labels == None:
                print("Points labels loading failed, got {}".format(lidar_seg_point_cloud.labels))
            else:
                print("Points labels shape: {}".format(lidar_seg_point_cloud.labels.shape))
                print("Points labels: \n{}".format(lidar_seg_point_cloud.labels))

    # return a tuple of points and labels
    return (lidar_seg_point_cloud.points, lidar_seg_point_cloud.labels)
