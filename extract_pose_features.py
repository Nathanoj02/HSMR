"""
Extract frozen SKEL pose parameters from HSMR model for XGBoost training.

This script:
1. Loads the pretrained HSMR model (frozen, no training)
2. Runs inference on your images
3. Extracts the 46-dimensional pose vector (q0-q45)
4. Saves features in a format ready for XGBoost

Usage:
    python extract_pose_features.py --input_path /path/to/images --output_path features.npz
"""

from lib.kits.basic import *
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

# Import detection and pipeline utilities
from lib.kits.hsmr_demo import build_detector, build_inference_pipeline, IMG_MEAN_255, IMG_STD_255, imgs_det2patches
from lib.platform.sliding_batches import asb


def load_images(input_path: Path):
    """Load images from a directory or single image."""
    if input_path.is_file():
        return [np.array(Image.open(input_path).convert('RGB'))], [input_path.name]
    elif input_path.is_dir():
        img_paths = sorted(list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')))
        images = [np.array(Image.open(p).convert('RGB')) for p in img_paths]
        names = [p.name for p in img_paths]
        return images, names
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def extract_poses_from_frames(
    frames: np.ndarray,
    detector,
    pipeline,
    device: str = 'cuda',
    rec_bs: int = 32,
    max_instances: int = 1,
) -> np.ndarray:
    """
    Extract SKEL pose parameters from video frames.

    Args:
        frames: (T, H, W, 3) array of RGB frames
        detector: Human detection model (from build_detector)
        pipeline: HSMR inference pipeline (from build_inference_pipeline)
        device: 'cuda' or 'cpu'
        rec_bs: Batch size for inference
        max_instances: Max people per frame (use 1 for single-person videos)

    Returns:
        poses: (T, 46) array of pose parameters for each frame
               Returns zeros if detection fails
    """
    # Detect humans in all frames
    detector_outputs = detector(frames)
    patches, det_meta = imgs_det2patches(frames, *detector_outputs, max_instances)

    if len(patches) == 0:
        # No person detected - return zeros
        return np.zeros((len(frames), 46), dtype=np.float32)

    # Extract pose parameters
    all_poses = []

    with torch.no_grad():
        for bw in asb(total=len(patches), bs_scope=rec_bs, enable_tqdm=False):
            patches_i = patches[bw.sid:bw.eid]

            # Normalize
            patches_normalized = (patches_i - IMG_MEAN_255) / IMG_STD_255
            patches_normalized = patches_normalized.transpose(0, 3, 1, 2)
            patches_normalized = torch.from_numpy(patches_normalized).float().to(device)

            # Inference
            outputs = pipeline(patches_normalized)
            poses = outputs['pd_params']['poses'].cpu().numpy()  # (B, 46)
            all_poses.append(poses)

    all_poses = np.concatenate(all_poses, axis=0)  # (N, 46)

    # Map instances back to frames (take first instance per frame)
    n_instances_per_frame = det_meta['n_patch_per_img']

    frame_poses = []
    idx = 0
    for n_inst in n_instances_per_frame:
        if n_inst > 0:
            frame_poses.append(all_poses[idx])  # Take first instance
            idx += n_inst
        else:
            # No detection in this frame - use zeros
            frame_poses.append(np.zeros(46, dtype=np.float32))

    return np.array(frame_poses)  # (T, 46)


def extract_pose_features(
    input_path: str,
    output_path: str,
    model_root: str = 'data_inputs/released_models/HSMR-ViTH-r1d1',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    det_bs: int = 8,
    rec_bs: int = 32,
    max_instances: int = 10,
):
    """
    Extract SKEL pose parameters from images using frozen HSMR model.

    Args:
        input_path: Path to image or directory of images
        output_path: Path to save extracted features (.npz or .npy)
        model_root: Path to pretrained HSMR model
        device: 'cuda' or 'cpu'
        det_bs: Batch size for detection
        rec_bs: Batch size for recovery/inference
        max_instances: Max number of people to detect per image

    Returns:
        Dictionary with extracted features:
            - 'poses': (N, 46) array of SKEL pose parameters
            - 'betas': (N, 10) array of SKEL shape parameters
            - 'cam_t': (N, 3) array of camera translations
            - 'image_ids': (N,) array of image identifiers
            - 'pose_names': List of 46 pose parameter names
    """

    print(f"Device: {device}")
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess images
    print(f"Loading images from {input_path}...")
    raw_imgs, img_names = load_images(input_path)
    print(f"\tFound {len(raw_imgs)} images")

    # 2. Detect humans in images
    print(f"Detecting humans...")
    detector = build_detector(
        batch_size=det_bs,
        max_img_size=1280,
        device=device,
    )
    detector_outputs = detector(raw_imgs)

    # Extract image patches around detected people
    patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, max_instances)
    if len(patches) == 0:
        raise RuntimeError("No human instances detected!")
    print(f"\tDetected {len(patches)} human instances")

    # 3. Load frozen HSMR model
    print(f"Loading pretrained HSMR model from {model_root}...")
    pipeline = build_inference_pipeline(model_root=model_root, device=device)
    pipeline.eval()  # Set to evaluation mode (frozen)
    print(f"\tModel loaded: {pipeline.name}")

    # 4. Extract pose features (FROZEN - no gradients)
    print(f"Extracting pose features with batch size {rec_bs}...")
    all_poses = []
    all_betas = []
    all_cam_t = []
    all_img_ids = []

    with torch.no_grad():  # No gradients needed - model is frozen
        for bw in asb(total=len(patches), bs_scope=rec_bs, enable_tqdm=True):
            # Get batch of patches
            patches_i = patches[bw.sid:bw.eid]  # (B, 256, 256, 3)

            # Normalize patches
            patches_normalized = (patches_i - IMG_MEAN_255) / IMG_STD_255
            patches_normalized = patches_normalized.transpose(0, 3, 1, 2)  # (B, 3, 256, 256)
            patches_normalized = torch.from_numpy(patches_normalized).float().to(device)

            # Run frozen inference
            outputs = pipeline(patches_normalized)

            # Extract predictions
            pd_params = outputs['pd_params']  # Dict with 'poses' and 'betas'
            poses = pd_params['poses'].cpu().numpy()      # (B, 46) - THE POSE ANGLES!
            betas = pd_params['betas'].cpu().numpy()      # (B, 10) - shape parameters
            cam_t = outputs['pd_cam_t'].cpu().numpy()     # (B, 3) - camera translation

            all_poses.append(poses)
            all_betas.append(betas)
            all_cam_t.append(cam_t)

            # Track which image each instance came from
            img_ids = np.repeat(
                np.arange(bw.sid, bw.eid),
                1
            )
            all_img_ids.append(img_ids)

    # 5. Concatenate all batches
    all_poses = np.concatenate(all_poses, axis=0)   # (N, 46)
    all_betas = np.concatenate(all_betas, axis=0)   # (N, 10)
    all_cam_t = np.concatenate(all_cam_t, axis=0)   # (N, 3)

    # Map instance indices to image names
    instance_to_img = []
    cur_idx = 0
    for i, n_instances in enumerate(det_meta['n_patch_per_img']):
        for j in range(n_instances):
            instance_to_img.append(img_names[i])
        cur_idx += n_instances

    print(f"Extracted features for {len(all_poses)} instances")
    print(f"\tPose features shape: {all_poses.shape}")
    print(f"\tShape features shape: {all_betas.shape}")

    # 6. Save features
    # Import pose parameter names for reference
    from thirdparty.SKEL.skel.kin_skel import pose_param_names

    features = {
        'poses': all_poses,              # (N, 46) - YOUR XGBOOST FEATURES!
        'betas': all_betas,              # (N, 10) - optional additional features
        'cam_t': all_cam_t,              # (N, 3) - camera params
        'image_names': np.array(instance_to_img),  # (N,) - image identifiers
        'pose_names': pose_param_names,  # List of 46 names (for reference)
    }

    print(f"Saving features to {output_path}...")
    np.savez_compressed(output_path, **features)

    print(f"Done! Features saved.")
    print(f"\nTo use with XGBoost:")
    print(f"  data = np.load('{output_path}')")
    print(f"  X = data['poses']  # Shape: (N, 46)")
    print(f"  # Add your labels (y) and train XGBoost")

    return features


def main():
    parser = argparse.ArgumentParser(description='Extract SKEL pose features for XGBoost')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--output_path', type=str, default='pose_features.npz',
                       help='Path to save extracted features')
    parser.add_argument('--model_root', type=str,
                       default='data_inputs/released_models/HSMR-ViTH-r1d1',
                       help='Path to pretrained HSMR model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--det_bs', type=int, default=8,
                       help='Detection batch size')
    parser.add_argument('--rec_bs', type=int, default=32,
                       help='Recovery batch size')
    parser.add_argument('--max_instances', type=int, default=10,
                       help='Max instances per image')

    args = parser.parse_args()

    extract_pose_features(
        input_path=args.input_path,
        output_path=args.output_path,
        model_root=args.model_root,
        device=args.device,
        det_bs=args.det_bs,
        rec_bs=args.rec_bs,
        max_instances=args.max_instances,
    )


if __name__ == '__main__':
    main()
