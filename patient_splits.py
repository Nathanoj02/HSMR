"""
Patient-based train/eval/test splits for mobility classification.
Runtime splits with random temporal cropping for generalization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Patient-to-label mapping
PATIENT_LABELS = {
    # Class 1: Mobility problems (A-E)
    'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,
    # Class 0: Non-problematic (F-J)
    'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0,
}

# Stratified patient splits (balanced across classes)
TRAIN_PATIENTS = ['A', 'B', 'C', 'F', 'G', 'H']  # 6 patients (3 per class)
EVAL_PATIENTS = ['D', 'I']                        # 2 patients (1 per class)
TEST_PATIENTS = ['E', 'J']                        # 2 patients (1 per class)

CLIP_DURATION = 1.5  # seconds


def get_patient_id(filename: str) -> str:
    """Extract patient ID from filename (e.g., 'A01.mp4' -> 'A')"""
    return Path(filename).name[0]


def get_video_files(dataset_dir: str = "dataset") -> List[Path]:
    """Get all video files from dataset directory"""
    dataset_path = Path(dataset_dir)
    return sorted(dataset_path.glob("*.mp4"))


def get_split(split: str, dataset_dir: str = "dataset") -> List[Dict]:
    """
    Get train, eval, or test split.

    Args:
        split: One of 'train', 'eval', 'test'
        dataset_dir: Path to dataset directory

    Returns:
        List of dicts with 'video_path', 'patient_id', 'label'
    """
    if split == 'train':
        patient_ids = TRAIN_PATIENTS
    elif split == 'eval':
        patient_ids = EVAL_PATIENTS
    elif split == 'test':
        patient_ids = TEST_PATIENTS
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'eval', or 'test'")

    video_files = get_video_files(dataset_dir)

    split_data = []
    for video_path in video_files:
        patient_id = get_patient_id(video_path.name)

        if patient_id in patient_ids:
            split_data.append({
                'video_path': str(video_path),
                'patient_id': patient_id,
                'label': PATIENT_LABELS[patient_id],
            })

    return split_data


def load_video_clip(video_path: str,
                    clip_duration: float = CLIP_DURATION,
                    random_start: bool = True) -> Tuple[np.ndarray, float]:
    """
    Load a random clip from video for generalization.

    Args:
        video_path: Path to video file
        clip_duration: Duration of clip in seconds (default 1.5s)
        random_start: If True, randomly select start time. If False, start from beginning.

    Returns:
        frames: numpy array of shape (num_frames, H, W, C)
        fps: frames per second of the video
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate clip parameters
    clip_frames = int(clip_duration * fps)

    # Handle videos shorter than clip duration
    if total_frames <= clip_frames:
        start_frame = 0
        clip_frames = total_frames
    else:
        if random_start:
            # Random start position
            max_start_frame = total_frames - clip_frames
            start_frame = np.random.randint(0, max_start_frame + 1)
        else:
            start_frame = 0

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read frames
    frames = []
    for _ in range(clip_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from video: {video_path}")

    return np.array(frames), fps


def get_train_data(dataset_dir: str = "dataset") -> List[Dict]:
    """Get training split (6 patients, 78 videos)"""
    return get_split('train', dataset_dir)


def get_eval_data(dataset_dir: str = "dataset") -> List[Dict]:
    """Get evaluation split (2 patients, 26 videos)"""
    return get_split('eval', dataset_dir)


def get_test_data(dataset_dir: str = "dataset") -> List[Dict]:
    """Get test split (2 patients, 26 videos)"""
    return get_split('test', dataset_dir)


def print_split_stats(dataset_dir: str = "dataset"):
    """Print statistics for all splits"""
    print("="*60)
    print("PATIENT-BASED SPLIT STATISTICS")
    print("="*60)

    for split_name in ['train', 'eval', 'test']:
        items = get_split(split_name, dataset_dir)
        n_total = len(items)
        n_class0 = sum(1 for x in items if x['label'] == 0)
        n_class1 = sum(1 for x in items if x['label'] == 1)
        patients = sorted(set(x['patient_id'] for x in items))

        print(f"\n{split_name.upper()}:")
        print(f"  Patients: {', '.join(patients)}")
        print(f"  Total videos: {n_total}")
        print(f"  Class 0 (non-problematic): {n_class0} videos")
        print(f"  Class 1 (mobility problems): {n_class1} videos")
        print(f"  Each video → random {CLIP_DURATION}s clip at runtime")

    print("="*60)


# Example usage
if __name__ == "__main__":
    # Print statistics
    print_split_stats()

    # Example: Load training data
    print("\nExample usage:")
    train_data = get_train_data()
    print(f"\nLoaded {len(train_data)} training videos")

    # Example: Load a random clip from first training video
    sample = train_data[0]
    print(f"\nLoading random 1.5s clip from: {sample['video_path']}")
    print(f"Patient: {sample['patient_id']}, Label: {sample['label']}")

    frames, fps = load_video_clip(sample['video_path'], random_start=True)
    print(f"Loaded {len(frames)} frames at {fps:.2f} fps")
    print(f"Clip shape: {frames.shape}")
