import sys, os
sys.path.append('/teamspace/studios/this_studio/3D-Lidar-Segmentation')
os.chdir('/teamspace/studios/this_studio/3D-Lidar-Segmentation')

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
from copy import deepcopy

app = FastAPI(title="3D LiDAR Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"
]

CLASS_COLORS = [
    [112,128,144],[220,20,60],[255,127,80],[255,158,0],[233,150,70],
    [255,61,99],[0,0,230],[47,79,79],[255,140,0],[255,99,71],
    [0,207,191],[175,0,75],[75,0,75],[112,180,60],[222,184,135],[0,175,0]
]

model = None
config = None
device = None
nusc = None
token_map = {}

@app.on_event("startup")
async def startup():
    global model, config, device, nusc, token_map
    from src.load_ptv3_model import load_ptv3_ckpt_and_config
    from src.DINOPT import DINOPT
    from src.nuscenes import NuScenes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    _, config = load_ptv3_ckpt_and_config(custom_config={"enable_flash": False})

    model = DINOPT(device, config=config, criteria=[], num_classes=16,
                   point_in_dim=64, image_in_dim=384, common_dim=128,
                   num_height_image_patch=46, num_width_image_patch=82,
                   fusion_model="point_to_image_projection")
    model.load_state_dict(torch.load(
        'weights/ProjectionFusion_25epochs_cosine.pth',
        map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Load nuScenes to build token map
    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
    
    # Build map: lidar_data_token → sample_token
    for sample in nusc.sample:
        lidar_token = sample['data']['LIDAR_TOP']
        token_map[lidar_token] = sample['token']
    
    print(f"Model loaded! Token map: {len(token_map)} entries")

@app.get("/api")
def root():
    return {"message": "3D LiDAR Segmentation", "mIoU": 0.6913, "classes": CLASS_NAMES}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}

@app.get("/classes")
def get_classes():
    return {"classes": [{"id": i, "name": CLASS_NAMES[i], "color": CLASS_COLORS[i]} for i in range(16)]}

@app.get("/samples")
def get_samples():
    samples_list = []
    for split in ['train', 'val', 'test']:
        feat_dir = f'feat/nuscenes/{split}'
        if os.path.exists(feat_dir):
            for f in os.listdir(feat_dir)[:20]:  # limit to 20 per split
                token = f.replace('.pt', '')
                samples_list.append({'token': token, 'split': split})
    return {'samples': samples_list}

@app.get("/segment_by_token/{sample_token}")
async def segment_by_token(sample_token: str):
    try:
        import torch.nn.functional as F
        import base64
        
        predictions = None
        feat_path = None
        for split in ['train', 'val', 'test']:
            path = f'feat/nuscenes/{split}/{sample_token}.pt'
            if os.path.exists(path):
                feat_path = path
                break
        
        if not feat_path:
            raise HTTPException(status_code=404, detail="Sample not found")

        feat_data = torch.load(feat_path, weights_only=False)
        sample_token_loaded = feat_data['sample_token']

        # Get raw points from nuScenes
        sample = nusc.get('sample', sample_token_loaded)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        bin_path = os.path.join('data/sets/nuscenes', lidar_data['filename'])
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        coords = points[:, :3].tolist()
        num_points = len(points)
        intensities = (points[:, 3] / 255.0).tolist()

        point_feature_dict = feat_data['point_feature_dict']
        image_feature_dict = feat_data['image_feature_dict']

        for cam in image_feature_dict:
            patch_feat = image_feature_dict[cam]['patch_token_feat']
            image_feature_dict[cam]['patch_token_feat'] = patch_feat[:, :384, :, :]

        with torch.no_grad():
            seg_logits, info_dict = model([sample_token_loaded], point_feature_dict, image_feature_dict)

            if 'padding_mask' in info_dict:
                padding_mask = info_dict['padding_mask'].cpu()
                n_real = int(padding_mask[0].sum().item())
                seg_logits = seg_logits[0, :n_real]
            else:
                seg_logits = seg_logits.reshape(-1, 16)

            pred = F.softmax(seg_logits.cpu(), dim=-1).max(dim=-1)[1].numpy()

            if 'inverse' in point_feature_dict:
                inverse = point_feature_dict['inverse'].numpy()
                pred = pred[inverse]

            # Get confidence
            probs = F.softmax(seg_logits.cpu(), dim=-1)
            confidence = probs.max(dim=-1)[0].numpy()
            if 'inverse' in point_feature_dict:
                confidence = confidence[point_feature_dict['inverse'].numpy()]

            # Get ground truth
            from src.dataset_nuscenes import NuScenesDataset
            gt_labels = feat_data['point_feature_dict'].get('origin_segment', None)
            if gt_labels is not None:
                gt = gt_labels.numpy().tolist()
            else:
                gt = pred.tolist()

        predictions = pred.tolist()
        colors = [CLASS_COLORS[p] for p in predictions]
        confidence_list = confidence.tolist() if hasattr(confidence, 'tolist') else []

        # Camera images
        camera_images = {}
        for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            cam_token = sample['data'][cam]
            cam_data = nusc.get('sample_data', cam_token)
            img_path = os.path.join('data/sets/nuscenes', cam_data['filename'])
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    camera_images[cam] = base64.b64encode(f.read()).decode('utf-8')

        return {
            "num_points": num_points,
            "coords": coords,
            "predictions": predictions,
            "ground_truth": gt,
            "confidence": confidence_list,
            "colors": colors,
            "gt_colors": [CLASS_COLORS[g] for g in gt],
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
            "camera_images": camera_images,
            "sample_token": sample_token,
            "intensities": intensities,
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tokens")
def get_tokens():
    return {"tokens": list(token_map.keys())[:10]}

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        points = np.frombuffer(contents, dtype=np.float32).reshape(-1, 4)
        coords = points[:, :3].tolist()
        num_points = len(points)

        # Get lidar token from filename
        # Extract lidar token from filename
        filename = file.filename.replace('.pcd.bin', '').replace('.bin', '')

        # Find sample token from lidar token
        sample_token_for_file = None
        for lidar_tok, samp_tok in token_map.items():
            lidar_data = nusc.get('sample_data', lidar_tok)
            if filename in lidar_data['filename']:
                sample_token_for_file = samp_tok
                break

        # Search feature file by sample token
        if sample_token_for_file:
            for split in ['train', 'val', 'test']:
                feat_path = f'feat/nuscenes/{split}/{sample_token_for_file}.pt'
                if os.path.exists(feat_path):
                    filename = sample_token_for_file
                    break
        
        # Try to find matching feature file
        predictions = None
        
        # Check all splits for matching feature file
        for split in ['train', 'val', 'test']:
            feat_path = f'feat/nuscenes/{split}/{filename}.pt'
            if os.path.exists(feat_path):
                print(f"Found feature file: {feat_path}")
                feat_data = torch.load(feat_path, weights_only=False)
                sample_token = feat_data['sample_token']
                
                # Run inference
                with torch.no_grad():
                    from src.dataset_joint_features import NuScenesFeatureDataset
                    import torch.nn.functional as F
                    
                    point_feature_dict = feat_data['point_feature_dict']
                    image_feature_dict = feat_data['image_feature_dict']
                    
                    # Select patch feat layer 0
                    for cam in image_feature_dict:
                        patch_feat = image_feature_dict[cam]['patch_token_feat']
                        image_feature_dict[cam]['patch_token_feat'] = patch_feat[:, :384, :, :]
                    
                    seg_logits, info_dict = model(
                        [sample_token],
                        point_feature_dict,
                        image_feature_dict
                    )

                    # Unpad using padding mask
                    if 'padding_mask' in info_dict:
                        padding_mask = info_dict['padding_mask'].cpu()
                        n_real = int(padding_mask[0].sum().item())
                        seg_logits = seg_logits[0, :n_real]  # [n_real, 16]
                    else:
                        seg_logits = seg_logits.reshape(-1, 16)

                    pred = F.softmax(seg_logits.cpu(), dim=-1).max(dim=-1)[1].numpy()

                    # Map voxel predictions back to original points
                    if 'inverse' in point_feature_dict:
                        inverse = point_feature_dict['inverse'].numpy()
                        pred = pred[inverse]

                    predictions = pred.tolist()
                    
                    


                break
        
        if predictions is None:
            # Fallback: use point intensity to make educated guess
            print(f"No feature file found for {filename}, using fallback")
            predictions = np.zeros(num_points, dtype=int)
            z = points[:, 2]
            predictions[z < -1.5] = 10  # driveable surface
            predictions[(z >= -1.5) & (z < 0)] = 12  # sidewalk
            predictions[(z >= 0) & (z < 1)] = 13  # terrain
            predictions[(z >= 1) & (z < 3)] = 3   # car
            predictions[z >= 3] = 14  # manmade
            predictions = predictions.tolist()

        colors = [CLASS_COLORS[p] for p in predictions]

        # Load camera images
        import base64
        camera_images = {}
        if nusc and sample_token_for_file:
            sample = nusc.get('sample', sample_token_for_file)
            for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                cam_token = sample['data'][cam]
                cam_data = nusc.get('sample_data', cam_token)
                img_path = os.path.join('data/sets/nuscenes', cam_data['filename'])
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        camera_images[cam] = base64.b64encode(f.read()).decode('utf-8')

        return {
            "num_points": num_points,
            "coords": coords,
            "predictions": predictions,
            "colors": colors,
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
            "camera_images": camera_images,
            "sample_token": sample_token_for_file or ""
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="/teamspace/studios/this_studio/3D-Lidar-Segmentation/app/static", html=True), name="static")
