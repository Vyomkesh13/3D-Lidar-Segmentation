import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import matplotlib.pyplot as plt

from src.nuscenes.utils.color_map import get_colormap

from dataset_nuscenes import NuScenesDataset

from sklearn.decomposition import PCA

####################### Linear Segmentation Head Visualization #######################
def visualize_seg_head(seg_head):
    # (16, 64)
    weights = seg_head.weight.cpu().detach().numpy()
    # (64, 16)
    weights = weights.T
    print("weights shape: {}".format(weights.shape))

    plt.imshow(weights, cmap='viridis')
    plt.colorbar()
    plt.title('Linear Segmentation Head Weights Heatmap')
    plt.xlabel('Output Classes')
    plt.ylabel('Input Features')
    plt.show()

    return weights

####################### DINOv2 PCA Visualization #######################
def visualize_pca(patch_tokens, title="Patch Embeddings (PCA → RGB)"):
    # PCA: 3 principal directions of variance in the feature space
    pca = PCA(n_components=3)
    patch_pca = pca.fit_transform(patch_tokens.cpu().numpy())

    # Normalize PCA components to [0, 1] for RGB display
    patch_rgb = (patch_pca - patch_pca.min(0)) / (patch_pca.max(0) - patch_pca.min(0))

    # Reshape to image grid (46, 82, 3)
    patch_rgb_img = patch_rgb.reshape(46, 82, 3)

    # Show as image
    plt.figure(figsize=(6, 6))
    plt.imshow(patch_rgb_img)
    plt.axis('off')
    plt.title(title)
    plt.show()

####################### Point Cloud 3D Visualization #######################
def visualize_point_cloud_segmentation(nusc, point_coord, point_segment, num_classes=16):

    nusc_colormap = get_colormap()
    learning_idx_to_lidarseg_idx = [NuScenesDataset.get_reversed_learning_map(-1)[seg][0] for seg in range(num_classes)]
    class_color = [nusc_colormap[nusc.lidarseg_idx2name_mapping[seg]] for seg in learning_idx_to_lidarseg_idx]

    color = np.array(class_color)[point_segment]

    pcd = o3d.geometry.PointCloud()
    if isinstance(point_coord, torch.Tensor):
        point_coord = point_coord.cpu().detach().numpy()

    pcd.points = o3d.utility.Vector3dVector(point_coord)

    pcd.colors = o3d.utility.Vector3dVector(color / 255)
    o3d.visualization.draw_plotly([pcd])
    return pcd

####################### Class IOU Bar Graph #######################
def plot_class_iou_bar_graph(config, class_based_record, title:str):
    # plot iou for each class - bar graph
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)

    class_based_record_sorted = dict(sorted(class_based_record.items(), key=lambda item: item[1]['iou'], reverse=True))

    class_iou = [0] * len(class_based_record_sorted)
    class_acc = [0] * len(class_based_record_sorted)
    idx = 0
    class_indices = []
    for class_idx, rec_dict in class_based_record_sorted.items():
        bar_graph = ax.bar(idx, round(rec_dict['iou'], 3), label=f"{class_idx+1}: {config.names[class_idx]}")
        ax.bar_label(bar_graph, padding=1)
        idx += 1
        class_indices.append(class_idx + 1)

    x = np.arange(len(config.names))
    ax.set_xticks(x, class_indices)
    ax.set_xlabel('Lidar Segmentation Index')
    ax.set_ylabel('IoU')
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.show() 

####################### PCA Visualization of Final Layer Point Features #######################
def visualize_pointcloud_pca(point_coord, point_features):
    # Perform PCA on features
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(point_features)

    # Normalize PCA components to [0, 1] for RGB display
    color = (features_pca - features_pca.min(0)) / (features_pca.max(0) - features_pca.min(0))

    pcd = o3d.geometry.PointCloud()
    if isinstance(point_coord, torch.Tensor):
        point_coord = point_coord.cpu().detach().numpy()

    pcd.points = o3d.utility.Vector3dVector(point_coord)

    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_plotly([pcd])