import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.model import PointTransformerV3, Point
from src.extract_features import CAMERA_TYPES
from src.model import offset2batch, batch2offset
from src.metrics import AverageMeter, intersection_and_union
from src.fusion_utils import pad_batch_voxels, project_point_to_pixel, get_pixel_patch, extract_associated_point_image_features

from src.dataset_nuscenes import get_point_data_by_sample_token

from src.loss import CrossEntropyLoss
from src.lovasz import LovaszLoss

LOSS_FUNCTIONS = {"CrossEntropyLoss": CrossEntropyLoss,
                  "LovaszLoss": LovaszLoss}

class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            loss_type = loss_cfg.pop("type")
            if loss_type in LOSS_FUNCTIONS:
                self.criteria.append(LOSS_FUNCTIONS[loss_type](**loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss
    
def get_num_points_per_class(sample_pred_results, num_classes):
    # Get the number of points per class (based on ground truth)
    num_points_per_class = np.array([0 for _ in range(num_classes)])
    for sampel_tok, result_dict in sample_pred_results.items():
        target = result_dict['sample_segment'].reshape(result_dict['sample_segment'].size)
        tok_num_points_per_class, _ = np.histogram(target, bins=np.arange(num_classes + 1))
        num_points_per_class += tok_num_points_per_class
    return num_points_per_class
    
FUSION_MODELS = ("direct", "point_to_image_projection", "ptv3_only")

class DINOPT(nn.Module):

    def __init__(self, device, 
                 config,
                 criteria,
                 num_classes:int, 
                 point_in_dim:int, 
                 image_in_dim:int, 
                 common_dim:int, 
                 num_height_image_patch:int,
                 num_width_image_patch:int,
                 fusion_model="direct",
                 residual=False):
        super(DINOPT, self).__init__()

        self.device = device
        self.config = config
        self.criteria = Criteria(criteria)

        self.num_classes = num_classes
        self.point_in_dim = point_in_dim
        self.image_in_dim = image_in_dim
        self.common_dim = common_dim
        self.num_height_image_patch = num_height_image_patch
        self.num_width_image_patch = num_width_image_patch
        self.fusion_model = fusion_model
        self.residual = residual

        assert self.fusion_model in FUSION_MODELS, f"Fusion model type unfound, expected to be in {FUSION_MODELS}"

        if fusion_model == "direct":
            # transform point features to common dimension
            self.point_transform = nn.Linear(point_in_dim, common_dim)
            # self.point_transform = nn.Identity()
            # transform image features to common dimension
            self.image_transform = nn.Linear(image_in_dim * 6, common_dim)

            # MLP to learn the combined the features
            self.model = nn.Sequential(
                nn.Linear(num_height_image_patch * num_width_image_patch, common_dim),
                nn.ReLU(),
                nn.Linear(common_dim, common_dim),
                nn.ReLU(),
            )

            self.segmentation_head = nn.Linear(common_dim, num_classes)

        elif fusion_model == "point_to_image_projection":
            # MLP to learn the combined the features
            self.model = nn.Sequential(
                nn.Linear(point_in_dim + image_in_dim, common_dim),
                nn.ReLU(),
                nn.Linear(common_dim, common_dim),
                nn.ReLU(),
            )

            self.segmentation_head = nn.Linear(common_dim, num_classes)

        elif fusion_model == "ptv3_only":
            self.segmentation_head = nn.Linear(point_in_dim, num_classes)
    
    def prep_input_direct(self, sample_token:str, point_feature_dict:dict, image_feature_dict:dict):
        ############### Point Feature Prep ###############
        # pad point features with zeros to have max_num_voxels per sample in batch
        padded_point_features, padding_mask = pad_batch_voxels(point_feature_dict, feat_key='feat')
        
        ############### Image Feature Prep ###############
        # image patch token feat output from dinov2 (batch_size, 384, num_height_patch, num_width_patch)
        image_features_list = []
        for camera in CAMERA_TYPES:
            # patch token feature of the camera image
            image_patch_tokens = image_feature_dict[camera]['patch_token_feat']
            batch_size, img_embed_dim, h_patch, w_patch = image_patch_tokens.shape

            # (batch_size, 384, num_patches)
            image_features = image_patch_tokens.reshape(batch_size, img_embed_dim, -1)
            # (batch_size, num_patches, 384)
            image_features = image_features.permute(0, 2, 1)

            image_features_list.append(image_features)
        # (batch_size, num_patches, 6 * 384)
        image_features = torch.cat(image_features_list, dim=-1)

        ############### Move Model Inputs to Cuda ###############
        padded_point_features = padded_point_features.to(self.device)
        image_features = image_features.to(self.device)

        return padded_point_features, image_features, padding_mask

    def forward_direct(self, point_features:torch.Tensor, image_features:torch.Tensor):
        """
        Predict segment label from the combined point and image features

        Args:
            point_features: point features learned by PTv3, shape (N, max_num_points, 64)
            image_features: image features learned by DINOv2, shape (N, 6, 4 * 384)
        """
        # transform point features (batch_size, max_num_voxels, common_dim)
        point_features = self.point_transform(point_features)
        # transform image features (batch_size, num_patches, common_dim)
        image_features = self.image_transform(image_features)

        # apply batch matrix multiply: dot product for every pair of point feature and image feature
        # (batch_size, max_num_voxels, num_patches)
        combo_features = torch.bmm(point_features, image_features.permute((0, 2, 1)))

        # apply model to combined features
        out_features = self.model(combo_features)

        # residual
        if self.residual:
            out_features = out_features + point_features.to(self.device)

        # segmentation logits (N)
        seg_logits = self.segmentation_head(out_features)

        return seg_logits, out_features.cpu()
    
    def forward_point_to_image_projection(self, sample_token:str, point_feature_dict:dict, image_feature_dict:dict):
        # pad point features with zeros to have max_num_voxels per sample in batch
        padded_point_features, padding_mask = pad_batch_voxels(point_feature_dict, feat_key='feat')

        cam_to_projections = project_point_to_pixel(sample_token, point_feature_dict, image_feature_dict)
        cam_to_projection_patch = get_pixel_patch(cam_to_projections, image_feature_dict)
        cam_to_associated_image_feature = extract_associated_point_image_features(sample_token, point_feature_dict, image_feature_dict, cam_to_projections, cam_to_projection_patch)

        # initialize combo features (batch_size, max_num_voxels, 64 + 384)
        combo_features = torch.cat((padded_point_features, torch.zeros_like(cam_to_associated_image_feature[CAMERA_TYPES[0]])), dim=-1)
        # iterate through each camera type and add the associated image patch feature
        for camera in CAMERA_TYPES:
            combo_features[:, :, self.point_in_dim:] += cam_to_associated_image_feature[camera]
        # take the average
        combo_features[:, :, self.point_in_dim:] = combo_features[:, :, self.point_in_dim:] / len(CAMERA_TYPES)
                    
        # move model input to cuda
        combo_features = combo_features.to(self.device)

        # apply model to combined features
        out_features = self.model(combo_features)

        # residual
        if self.residual:
            out_features = out_features + padded_point_features.to(self.device)

        # segmentation logits (N)
        seg_logits = self.segmentation_head(out_features)
        
        return seg_logits, padding_mask, out_features.cpu()

    def forward_ptv3_only(self, sample_token:str, point_feature_dict:dict):
        # (num_batch_voxels, 64)
        point_features = point_feature_dict['feat']

        point_features = point_features.to(self.device)

        seg_logits = self.segmentation_head(point_features)

        return seg_logits, point_features.cpu()

    def forward(self, sample_token:str, point_feature_dict:dict, image_feature_dict:dict, return_last_embed=False):
        seg_logits = None
        last_point_embed = None
        info_dict = {}
        
        if self.fusion_model == "direct":
            point_features, image_features, padding_mask = self.prep_input_direct(sample_token, point_feature_dict, image_feature_dict)
            seg_logits, last_point_embed = self.forward_direct(point_features, image_features)    
            info_dict['padding_mask'] = padding_mask    

        elif self.fusion_model == "point_to_image_projection":
            seg_logits, padding_mask, last_point_embed = self.forward_point_to_image_projection(sample_token, point_feature_dict, image_feature_dict)
            info_dict['padding_mask'] = padding_mask

        elif self.fusion_model == "ptv3_only":
            seg_logits, last_point_embed = self.forward_ptv3_only(sample_token, point_feature_dict)
            info_dict['last_point_embed'] = last_point_embed
        
        if return_last_embed:
                info_dict['last_point_embed'] = last_point_embed

        return seg_logits, info_dict
    
def dinopt_train_val(model:DINOPT, 
                     feature_dataloader:DataLoader, 
                     optimizer=None, 
                     curr_epoch=1, 
                     total_epochs=1, 
                     scheduler=None,
                     wandb_run=None,
                     return_pred_results=False,
                     return_last_embed=False):
    is_training = optimizer is not None

    model.train() if is_training else model.eval()
    
    # evaluation metrics
    batch_time_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # dictionary mapping sample token to a dictionary of evaluation results
    record = {}
    # dictionary mapping sample token to prediction results
    sample_pred_results = {}

    data_bar = tqdm(feature_dataloader)
    with (torch.enable_grad() if is_training else torch.no_grad()):
        for sample_token, point_feature_dict, image_feature_dict in data_bar:
            start_time = time.time()

            # obtain batch size (= number of samples)
            batch_size = len(sample_token)
            # ground truth segment labels of voxels
            segment = point_feature_dict['segment']

            ############### Run Model ###############
            seg_logits, info_dict = model(sample_token, point_feature_dict, image_feature_dict, return_last_embed)

            seg_logits = seg_logits.cpu()

            ############### Unpad Segmentation Logits ###############
            if 'padding_mask' in info_dict:                
                padding_mask = info_dict['padding_mask'].cpu()

                seg_logits_list = []
                for b in range(batch_size):
                    n_real = padding_mask[b].sum().item()
                    seg_logits_list.append(seg_logits[b, :n_real])
                seg_logits = torch.cat(seg_logits_list, dim=0)

                if 'last_point_embed' in info_dict:
                    last_point_embed = info_dict['last_point_embed'].cpu()

                    last_point_embed_list = []
                    for b in range(batch_size):
                        n_real = padding_mask[b].sum().item()
                        last_point_embed_list.append(last_point_embed[b, :n_real])
                    last_point_embed = torch.cat(last_point_embed_list, dim=0)
                
            assert seg_logits.shape[0] == segment.shape[0], f"seg logits shape: {seg_logits.shape}, segment shape: {segment.shape}"

            ############### Compute Loss and Backpropagation ###############
            loss = model.criteria(seg_logits.to(self.device), segment.to(self.device))

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler != None:
                    scheduler.step()

            # predicted segment on voxelized points
            pred = F.softmax(seg_logits, dim=-1)
            pred = pred.max(dim=-1)[1].cpu()

            # list of predicted segment for each sample in batch
            batch_pred_list = list(torch.tensor_split(pred, point_feature_dict['offset']))[:batch_size]
            # list of inverse vector for each sample in batch
            inverse_list = list(torch.tensor_split(point_feature_dict["inverse"], point_feature_dict['origin_offset']))[:batch_size]

            # predicted segment on original points
            if "origin_segment" in point_feature_dict.keys():
                assert "inverse" in point_feature_dict.keys()
                # predicted segment
                batch_pred_list_final = []
                for b in range(batch_size):
                    batch_pred_list_final.append(batch_pred_list[b][inverse_list[b]])
                pred = torch.cat(batch_pred_list_final, dim=0)
                # true segment
                segment = point_feature_dict["origin_segment"]

                # list of predicted segment for each sample in batch
                batch_pred_list = list(torch.tensor_split(pred, point_feature_dict['origin_offset']))[:batch_size]
                batch_segment_list = list(torch.tensor_split(segment, point_feature_dict['origin_offset']))[:batch_size]
            else:
                batch_segment_list = list(torch.tensor_split(segment, point_feature_dict['offset']))[:batch_size]

            assert len(batch_pred_list) == batch_size and len(batch_segment_list) == batch_size 

            # evaluation metrics
            b_m_iou = 0
            b_m_acc = 0
            for sample_tok, sample_pred, sample_segment in zip(sample_token, batch_pred_list, batch_segment_list):
                intersection, union, target = intersection_and_union(sample_pred.cpu().numpy(), sample_segment.cpu().numpy(), K=model.num_classes, ignore_index=-1)
                # update meter
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)

                # update record
                record[sample_tok] = dict(intersection=intersection, union=union, target=target)

                mask = union != 0
                iou_class = intersection / (union + 1e-10)
                iou = np.mean(iou_class[mask])
                acc = sum(intersection) / (sum(target) + 1e-10)

                b_m_iou += iou
                b_m_acc += acc
            # average iou of training samples in current batch
            b_m_iou = b_m_iou / batch_size
            # average accuracy of training samples in current batch
            b_m_acc = b_m_acc / batch_size

            batch_train_duration = time.time() - start_time
            batch_time_meter.update(batch_train_duration)

            data_bar.set_description("Epoch: {}/{} "
                                        "Loss: {} "
                                        "Batch mean accuracy {b_m_acc:.4f} "
                                        "Batch mIoU {b_m_iou:.4f} ".format(
                                            curr_epoch,
                                            total_epochs,
                                            loss.data,
                                            acc=acc,
                                            b_m_acc=b_m_acc,
                                            b_m_iou=b_m_iou,
                                        ))
            
            if wandb_run != None:
                wandb_run.log({"loss": loss.data,
                               "mean accuracy": b_m_acc,
                               "mIoU": b_m_iou,
                               "duration (sec)": batch_train_duration})
            
            if return_pred_results:
                if return_last_embed:
                    last_point_embed_list = list(torch.tensor_split(last_point_embed, point_feature_dict['offset']))[:batch_size]

                for b, (sample_tok, sample_pred, sample_segment) in enumerate(zip(sample_token, batch_pred_list, batch_segment_list)):
                    sample_pred_results[sample_tok] = dict(sample_pred=sample_pred.cpu().numpy(), 
                                                            sample_segment=sample_segment.cpu().numpy(),
                                                            b_m_iou=b_m_iou,
                                                            b_m_acc=b_m_acc)
                    if return_last_embed:
                        sample_pred_results[sample_tok]['last_point_embed'] = last_point_embed_list[b][inverse_list[b]].cpu()
                        
            ####################
            #   Remove Break   #
            ####################
            # break

        ########## validation only ##########
        class_based_record = {}
        freqweighted_iou = None
        if not is_training:
            # Get the number of points per class (based on ground truth)
            num_points_per_class = get_num_points_per_class(sample_pred_results, num_classes=model.config.data.num_classes)

            ########################## Mean IOU and Mean Accuracy ##########################
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)

            # modify IoU per class based on whether there are points in each class
            iou_class_modified = iou_class[num_points_per_class != 0]
            # modify accuracy per class based on whether there are points in each class
            accuracy_class_modified = accuracy_class[num_points_per_class != 0]

            mIoU = np.mean(iou_class_modified)
            mAcc = np.mean(accuracy_class_modified)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            print(
                "Evaluation result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )

            class_based_record = {}
            iou_per_class = []
            for i in range(model.num_classes):
                num_pts_cls = num_points_per_class[i]
                class_based_record[i] = dict(iou=iou_class[i], accuracy=accuracy_class[i])
                iou_per_class.append(iou_class[i])

                print(
                    "Class_{idx} - num gt points: {num_pts_cls} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        num_pts_cls=num_pts_cls,
                        name=model.config.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            
            ########################## Frequency-Weighted IOU Over the Classes ##########################
            # # Get the total number of points in the eval set.
            num_points_total = num_points_per_class.sum()

            # Get the IOU per class.
            # iou_per_class = get_per_class_iou()
            iou_per_class = np.array(iou_per_class)
            # Weight the IOU by frequency and sum across the classes.
            freqweighted_iou = float(np.nansum(num_points_per_class * iou_per_class) / num_points_total)

    return sample_pred_results, class_based_record, freqweighted_iou