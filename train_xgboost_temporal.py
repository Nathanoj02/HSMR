"""
Train XGBoost for mobility classification using temporal pose statistics.

This script:
1. Loads videos from patient-based splits
2. Extracts SKEL pose parameters (q) for all frames in each video clip
3. Computes temporal statistics (mean, std, min, max) per joint
4. Trains XGBoost on these aggregated features

Usage:
    python train_xgboost_temporal.py --output_model mobility_classifier.json
"""

from lib.kits.hsmr_demo import *

# Import patient splits
from patient_splits import get_train_data, get_eval_data, get_test_data, load_video_clip

# Import pose extraction from extract_pose_features
from extract_pose_features import extract_poses_from_frames


def compute_temporal_statistics(poses: np.ndarray) -> np.ndarray:
    """
    Compute temporal statistics over pose sequences.

    Args:
        poses: (T, 46) array of pose parameters across T frames

    Returns:
        features: (184,) feature vector with:
                  - mean (46), std (46), min (46), max (46)
                  = 184 total features
    """
    if len(poses) == 0 or poses.sum() == 0:
        # No valid poses - return zeros
        return np.zeros(184, dtype=np.float32)

    # Compute statistics per joint
    mean_poses = np.mean(poses, axis=0)  # (46,)
    std_poses = np.std(poses, axis=0)    # (46,)
    min_poses = np.min(poses, axis=0)    # (46,)
    max_poses = np.max(poses, axis=0)    # (46,)

    # Concatenate all statistics
    features = np.concatenate([mean_poses, std_poses, min_poses, max_poses])  # (184,)

    return features


def extract_features_from_video(
    video_path: str,
    detector,
    pipeline,
    device: str = 'cuda',
    clip_duration: float = 1.5,
    random_start: bool = True,
) -> np.ndarray:
    """
    Extract temporal pose features from a video.

    Args:
        video_path: Path to video file
        detector: Human detector
        pipeline: HSMR pipeline
        device: Device for inference
        clip_duration: Duration of clip to extract
        random_start: Whether to randomly sample clip

    Returns:
        features: (184,) temporal statistics feature vector
    """
    # Load video clip
    try:
        frames, fps = load_video_clip(video_path, clip_duration, random_start)
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return np.zeros(184, dtype=np.float32)

    # Extract poses for all frames
    poses = extract_poses_from_frames(frames, detector, pipeline, device)

    # Compute temporal statistics
    features = compute_temporal_statistics(poses)

    return features


def build_dataset(
    split_name: str,
    detector,
    pipeline,
    device: str = 'cuda',
    dataset_dir: str = 'dataset',
    clip_duration: float = 1.5,
    random_start: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and labels for a data split.

    Args:
        split_name: 'train', 'eval', or 'test'
        detector: Human detector
        pipeline: HSMR pipeline
        device: Device for inference
        dataset_dir: Dataset directory
        clip_duration: Clip duration in seconds
        random_start: Random temporal cropping

    Returns:
        X: (N, 184) feature matrix
        y: (N,) label vector
    """
    # Get split data
    if split_name == 'train':
        data = get_train_data(dataset_dir)
    elif split_name == 'eval':
        data = get_eval_data(dataset_dir)
    elif split_name == 'test':
        data = get_test_data(dataset_dir)
    else:
        raise ValueError(f"Invalid split: {split_name}")

    print(f"\nBuilding {split_name} dataset ({len(data)} videos)...")

    X = []
    y = []

    for item in tqdm(data, desc=f"Processing {split_name}"):
        # Extract features
        features = extract_features_from_video(
            item['video_path'],
            detector,
            pipeline,
            device,
            clip_duration,
            random_start,
        )

        X.append(features)
        y.append(item['label'])

    X = np.array(X)  # (N, 184)
    y = np.array(y)  # (N,)

    print(f"{split_name}: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

    return X, y


def train_xgboost_mobility_classifier(
    dataset_dir: str = 'dataset',
    output_path: str = 'mobility_classifier.json',
    model_root: str = 'data_inputs/released_models/HSMR-ViTH-r1d1',
    device: str = 'cuda',
    clip_duration: float = 1.5,
    random_training_clips: bool = True,
):
    """
    Train XGBoost classifier for mobility problem detection.

    Args:
        dataset_dir: Directory containing patient videos
        output_path: Path to save trained model
        model_root: Path to HSMR pretrained model
        device: 'cuda' or 'cpu'
        clip_duration: Duration of video clips to use
        random_training_clips: Use random clips during training (data augmentation)
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError:
        print("Missing dependencies. Install with:")
        print("pip install xgboost scikit-learn")
        return

    print("="*60)
    print("MOBILITY CLASSIFICATION WITH TEMPORAL POSE FEATURES")
    print("="*60)
    print(f"Device: {device}")
    print(f"Clip duration: {clip_duration}s")
    print(f"Random training clips: {random_training_clips}")

    # 1. Load models
    print(f"\nLoading human detector...")
    detector = build_detector(batch_size=8, max_img_size=1280, device=device)

    print(f"Loading HSMR model from {model_root}...")
    pipeline = build_inference_pipeline(model_root=model_root, device=device)
    pipeline.eval()
    print(f"\tModel loaded: {pipeline.name}")

    # 2. Build datasets
    print(f"\nBuilding datasets...")

    X_train, y_train = build_dataset(
        'train', detector, pipeline, device, dataset_dir,
        clip_duration, random_start=random_training_clips
    )

    X_eval, y_eval = build_dataset(
        'eval', detector, pipeline, device, dataset_dir,
        clip_duration, random_start=False  # Always use full clips for eval
    )

    X_test, y_test = build_dataset(
        'test', detector, pipeline, device, dataset_dir,
        clip_duration, random_start=False  # Always use full clips for test
    )

    # 3. Train XGBoost
    print(f"\nTraining XGBoost classifier...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'logloss',
        'seed': 42,
    }

    evals = [(dtrain, 'train'), (deval, 'eval')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10,
    )

    # 4. Evaluate on test set
    print(f"\nEvaluating on test set...")
    dtest = xgb.DMatrix(X_test, label=y_test)

    y_test_pred_prob = bst.predict(dtest)
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)

    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Non-problematic', 'Mobility Problems']))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"              Predicted")
    print(f"              0     1")
    print(f"Actual 0   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       1   {cm[1,0]:4d}  {cm[1,1]:4d}")

    # 5. Feature importance
    print(f"\nTop 10 most important features:")
    importance = bst.get_score(importance_type='weight')
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    from thirdparty.SKEL.skel.kin_skel import pose_param_names

    stat_names = ['mean', 'std', 'min', 'max']

    for feat_idx, score in sorted_features:
        feat_num = int(feat_idx.replace('f', ''))
        stat_type = stat_names[feat_num // 46]
        joint_idx = feat_num % 46
        joint_name = pose_param_names[joint_idx]
        print(f"   {stat_type}_{joint_name:30s} (f{feat_num:03d}): {score:.1f}")

    # 6. Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(output_path))
    print(f"\nModel saved to {output_path}")
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    return bst


def main():
    parser = argparse.ArgumentParser(
        description='Train XGBoost for mobility classification using temporal pose features'
    )
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                       help='Directory containing patient videos')
    parser.add_argument('--output_model', type=str, default='mobility_classifier.json',
                       help='Path to save trained model')
    parser.add_argument('--model_root', type=str,
                       default='data_inputs/released_models/HSMR-ViTH-r1d1',
                       help='Path to HSMR pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    parser.add_argument('--clip_duration', type=float, default=1.5,
                       help='Video clip duration in seconds')
    parser.add_argument('--no_random_clips', action='store_true',
                       help='Disable random temporal cropping during training')

    args = parser.parse_args()

    train_xgboost_mobility_classifier(
        dataset_dir=args.dataset_dir,
        output_path=args.output_model,
        model_root=args.model_root,
        device=args.device,
        clip_duration=args.clip_duration,
        random_training_clips=not args.no_random_clips,
    )


if __name__ == '__main__':
    main()
