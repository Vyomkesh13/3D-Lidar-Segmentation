import os
import time
import torch
import numpy as np

from src.utils import collate_fn
import torch.nn.functional as F
from src.nuscenes.eval.lidarseg.utils import ConfusionMatrix

from src.metrics import AverageMeter, intersection_and_union

def test_ptv3(model, 
                config, 
                test_dataloader,
                save_path='exp/nuscenes/semseg-pt-v3m1-0-base', 
                use_saved=False):
    model.eval()

    # evaluation metrics
    batch_time_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    # path to save model test result
    save_path = os.path.join(config.save_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # dictionary mapping lidar sample data token to a dictionary of evaluation results
    record = {}
    # dictionary mapping lidar sample data token to predicted segment and ground truth segment
    pred_and_gt = {}
    # fragment inference
    for idx, (sample_token, point_data_dict, camera_data_dict) in enumerate(test_dataloader):
        start_time = time.time()

        # assume batch size of 1
        point_data_dict = point_data_dict[0]
        fragment_list = point_data_dict.pop("fragment_list")
        segment = point_data_dict.pop("segment")

        ### investigate ###
        # print(sample_token)
        # print(f"what is in point_data_dict: {point_data_dict.keys()}")
        # print(f"original segment shape: {point_data_dict['origin_segment'].shape}")
        # print(f"segment shape: {segment.shape}")
        # print(f"invserse shape: {point_data_dict['inverse'].shape}")

        # print()
        # print(f"length of fragment list: {len(fragment_list)}")
        # print(f"what is in each fragment: {fragment_list[0].keys()}")
        # print(f"segment coord shape: {fragment_list[0]['coord'].shape}")
        # print(f"segment feat shape: {fragment_list[0]['feat'].shape}")


        lidar_data_token = point_data_dict.pop("name")
        pred_save_path = os.path.join(save_path, "{}_pred.npy".format(lidar_data_token))
        if use_saved and os.path.isfile(pred_save_path):
            # load saved prediction
            pred = np.load(pred_save_path)
            if "origin_segment" in point_data_dict.keys():
                segment = point_data_dict["origin_segment"]

            print("{}/{}: {}, loaded pred and label.".format(
                idx + 1, len(test_dataloader), lidar_data_token))
        else:
            # initialize prediction with shape (N, C)
            pred = torch.zeros((segment.size, config.data.num_classes)).cuda()
            # iterate through each fragment
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                # move the tensors in fragment's point input dict to cuda
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                
                # run fragment point input dict through model
                with torch.no_grad():
                    pred_part = model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)

                    if config.empty_cache:
                        torch.cuda.empty_cache()

                    # update prediction with seg logits according to offset
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

            # predict final segment class from logits
            pred = pred.max(1)[1].data.cpu().numpy()

            if "origin_segment" in point_data_dict.keys():
                assert "inverse" in point_data_dict.keys()
                # predicted segment
                pred = pred[point_data_dict["inverse"]]
                # true segment
                segment = point_data_dict["origin_segment"]

            # save predictions to file
            np.save(pred_save_path, pred)

        # update dictionary
        pred_and_gt[lidar_data_token] = dict(pred=pred, segment=segment)
        
        # evaluation metrics
        intersection, union, target = intersection_and_union(pred, segment, config.data.num_classes, config.data.ignore_index)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        # update record dictionary
        record[lidar_data_token] = dict(
            intersection=intersection, union=union, target=target
        )

        mask = union != 0
        iou_class = intersection / (union + 1e-10)
        iou = np.mean(iou_class[mask])
        acc = sum(intersection) / (sum(target) + 1e-10)

        m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
        m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

        batch_time_meter.update(time.time() - start_time)

        eval_msg = ("Test: {} [{}/{}]-{} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Accuracy {acc:.4f} ({m_acc:.4f}) "
                    "mIoU {iou:.4f} ({m_iou:.4f})".format(
                        lidar_data_token,
                        idx + 1,
                        len(test_dataloader),
                        segment.size,
                        batch_time=batch_time_meter,
                        acc=acc,
                        m_acc=m_acc,
                        iou=iou,
                        m_iou=m_iou,
                    ))

        print(eval_msg)

        ### investigate ###
        # break


    class_based_record = {}
    freqweighted_iou = 0
    ########################## Mean IOU and Mean Accuracy ##########################
    record_sync = [record]
    record = {}
    for _ in range(len(record_sync)):
        r = record_sync.pop()
        record.update(r)
        del r

    intersection = np.sum(
        [meters["intersection"] for _, meters in record.items()], axis=0
    )
    union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
    target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
    
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    
    print(
        "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
            mIoU, mAcc, allAcc
        )
    )

    class_based_record = {}
    iou_per_class = []
    for i in range(config.data.num_classes):
        class_based_record[i] = dict(iou=iou_class[i], accuracy=accuracy_class[i])
        iou_per_class.append(iou_class[i])

        print(
            "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                idx=i,
                name=config.data.names[i],
                iou=iou_class[i],
                accuracy=accuracy_class[i],
            )
        )

    ########################## Frequency-Weighted IOU Over the Classes ##########################
    # Get the number of points per class (based on ground truth)
    num_points_per_class = np.array([0 for _ in range(config.data.num_classes)])
    for tok, rec_dict in pred_and_gt.items():
        target = rec_dict['segment'].reshape(rec_dict['segment'].size)
        tok_num_points_per_class, _ = np.histogram(target, bins=np.arange(config.data.num_classes + 1))
        num_points_per_class += tok_num_points_per_class

    # # Get the total number of points in the eval set.
    num_points_total = num_points_per_class.sum()

    # Get the IOU per class.
    # iou_per_class = get_per_class_iou()
    iou_per_class = np.array(iou_per_class)
    # Weight the IOU by frequency and sum across the classes.
    freqweighted_iou = float(np.nansum(num_points_per_class * iou_per_class) / num_points_total)
    
    return record, pred_and_gt, class_based_record, freqweighted_iou




def val_ptv3(model, 
                config, 
                test_dataloader,
                save_path='exp/nuscenes/semseg-pt-v3m1-0-base', 
                use_saved=False):
    model.eval()

    # evaluation metrics
    batch_time_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    # path to save model test result
    save_path = os.path.join(config.save_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # dictionary mapping lidar sample data token to a dictionary of evaluation results
    record = {}
    # dictionary mapping lidar sample data token to predicted segment and ground truth segment
    pred_and_gt = {}
    # fragment inference
    for idx, (sample_token, point_data_dict, camera_data_dict) in enumerate(test_dataloader):
        start_time = time.time()

        inference_fragments = "fragment_list" in point_data_dict
        if inference_fragments:
            fragment_list = point_data_dict.pop("fragment_list")
        segment = point_data_dict["segment"].numpy()

        ### investigate ###
        # print(sample_token)
        # print(f"what is in point_data_dict: {point_data_dict.keys()}")
        # print(f"original segment shape: {point_data_dict['origin_segment'].shape}")
        # print(f"segment shape: {segment.shape}")
        # print(f"invserse shape: {point_data_dict['inverse'].shape}")

        # print()
        # print(f"length of fragment list: {len(fragment_list)}")
        # print(f"what is in each fragment: {fragment_list[0].keys()}")
        # print(f"segment coord shape: {fragment_list[0]['coord'].shape}")
        # print(f"segment feat shape: {fragment_list[0]['feat'].shape}")

        loss = None
        sample_token = sample_token[0]
        pred_save_path = os.path.join(save_path, "{}_pred.npy".format(sample_token))
        if use_saved and os.path.isfile(pred_save_path):
            # load saved prediction
            pred = np.load(pred_save_path)
            if "origin_segment" in point_data_dict.keys():
                segment = point_data_dict["origin_segment"]

            print("{}/{}: {}, loaded pred and label.".format(
                idx + 1, len(test_dataloader), sample_token))
        else:
            # initialize prediction with shape (N, C)
            pred = torch.zeros((segment.size, config.data.num_classes)).cuda()
            
            if not inference_fragments:
                input_dict = collate_fn([point_data_dict])
                # move the tensors in fragment's point input dict to cuda
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                with torch.no_grad():
                    pred = model(input_dict)
                    loss = pred['loss']
                    pred = pred["seg_logits"]  # (n, k)
                    pred = F.softmax(pred, -1)

                    if config.empty_cache:
                        torch.cuda.empty_cache()
            else:
                # iterate through each fragment
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    # move the tensors in fragment's point input dict to cuda
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    
                    # run fragment point input dict through model
                    with torch.no_grad():
                        pred_part = model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)

                        if config.empty_cache:
                            torch.cuda.empty_cache()

                        # update prediction with seg logits according to offset
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

            # predict final segment class from logits
            pred = pred.max(1)[1].data.cpu().numpy()

            if "origin_segment" in point_data_dict.keys():
                assert "inverse" in point_data_dict.keys()
                # predicted segment
                pred = pred[point_data_dict["inverse"]]
                # true segment
                segment = point_data_dict["origin_segment"]
                if isinstance(segment, torch.Tensor):
                    segment = segment.numpy()

            # save predictions to file
            np.save(pred_save_path, pred)

        # update dictionary
        pred_and_gt[sample_token] = dict(pred=pred, segment=segment, loss=loss)
        
        # evaluation metrics
        intersection, union, target = intersection_and_union(pred, segment, config.data.num_classes, config.data.ignore_index)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        # update record dictionary
        record[sample_token] = dict(
            intersection=intersection, union=union, target=target
        )

        mask = union != 0
        iou_class = intersection / (union + 1e-10)
        iou = np.mean(iou_class[mask])
        acc = sum(intersection) / (sum(target) + 1e-10)

        m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
        m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

        batch_time_meter.update(time.time() - start_time)

        eval_msg = ("Test: {} [{}/{}]-{} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Accuracy {acc:.4f} ({m_acc:.4f}) "
                    "mIoU {iou:.4f} ({m_iou:.4f})".format(
                        sample_token,
                        idx + 1,
                        len(test_dataloader),
                        segment.size,
                        batch_time=batch_time_meter,
                        acc=acc,
                        m_acc=m_acc,
                        iou=iou,
                        m_iou=m_iou,
                    ))

        print(eval_msg)

        ### investigate ###
        # break


    class_based_record = {}
    freqweighted_iou = 0
    ########################## Mean IOU and Mean Accuracy ##########################
    # record_sync = [record]
    # record = {}
    # for _ in range(len(record_sync)):
    #     r = record_sync.pop()
    #     record.update(r)
    #     del r

    intersection = np.sum(
        [meters["intersection"] for _, meters in record.items()], axis=0
    )
    union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
    target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
    
    mask = union != 0

    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class[mask])
    mAcc = np.mean(accuracy_class[mask])
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    
    print(
        "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
            mIoU, mAcc, allAcc
        )
    )

    class_based_record = {}
    iou_per_class = []
    for i in range(config.data.num_classes):
        class_based_record[i] = dict(iou=iou_class[i], accuracy=accuracy_class[i])
        iou_per_class.append(iou_class[i])

        print(
            "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                idx=i,
                name=config.data.names[i],
                iou=iou_class[i],
                accuracy=accuracy_class[i],
            )
        )

    ########################## Frequency-Weighted IOU Over the Classes ##########################
    # Get the number of points per class (based on ground truth)
    num_points_per_class = np.array([0 for _ in range(config.data.num_classes)])
    for tok, rec_dict in pred_and_gt.items():
        target = rec_dict['segment'].reshape(rec_dict['segment'].size)
        tok_num_points_per_class, _ = np.histogram(target, bins=np.arange(config.data.num_classes + 1))
        num_points_per_class += tok_num_points_per_class

    # # Get the total number of points in the eval set.
    num_points_total = num_points_per_class.sum()

    # Get the IOU per class.
    # iou_per_class = get_per_class_iou()
    iou_per_class = np.array(iou_per_class)
    # Weight the IOU by frequency and sum across the classes.
    freqweighted_iou = float(np.nansum(num_points_per_class * iou_per_class) / num_points_total)
    
    return record, pred_and_gt, class_based_record, freqweighted_iou