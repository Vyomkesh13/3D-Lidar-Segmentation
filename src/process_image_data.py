import os
import numpy as np
from typing import List, Dict, Any, Tuple
from pyquaternion import Quaternion

from src.nuscenes import NuScenes


def obtain_sensor2top(nusc:NuScenes,
                        sensor_token,
                        l2e_t,
                        l2e_r_mat,
                        e2g_t,
                        e2g_r_mat):
    """
    Obtain the info with RT matricies from general sensor to Top LiDAR.

    Args:
        nusc (class): NuScenes instance.
        sensor_token (str): Sample data token corresponding to the specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global in shape (3, 3).

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sample_data_record = nusc.get('sample_data', sensor_token)
    calibrated_sensor_record = nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])

    pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sample_data_record['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
        
    sweep = {
        'data_path': data_path,
        'sample_data_token': sample_data_record['token'],
        'sensor2ego_translation': calibrated_sensor_record['translation'],
        'sensor2ego_rotation': calibrated_sensor_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def get_sample_camera_images(nusc:NuScenes, sample:Dict[str, Any]):
    # lidar token and sample data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sample_data_record = nusc.get('sample_data', lidar_token)
    # definition of lidar sensor as calibrated on a particular vehicle
    calib_rec = nusc.get('calibrated_sensor', lidar_sample_data_record['calibrated_sensor_token'])
    # ego vehicle pose
    pose_rec = nusc.get('ego_pose', lidar_sample_data_record['ego_pose_token'])

    # lidar to ego rotation
    l2e_r = calib_rec['rotation']
    # lidar to ego translation
    l2e_t = calib_rec['translation']
    # ego to global rotation
    e2g_r = pose_rec['rotation']
    # ego to global translation
    e2g_t = pose_rec['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    
    # dictionary mapping camera type to camera info
    camera_dict = {}
    # 6 cameras
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    # iterate through each camera and obtain the camera information for the sample
    for cam in camera_types:
        # camera token
        cam_token = sample['data'][cam]
        # obtain camera image filepath and camera intrinsic
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        # obtain info with rotation and translation matrics from camera to top lidar
        cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat)
        cam_info.update(cam_intrinsic=cam_intrinsic)
        # update the output camera data dictionary
        camera_dict[cam] = cam_info
    return camera_dict
