"""Historical offline preprocessing; see docs/SCIENTIFIC_AUDIT.md before use.

The CSV header behavior, 25 Hz filter assumption, label-conditioned peaks,
whole-group normalization and multimodal DTW are deliberately preserved.
"""

import glob
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import euclidean

WINDOW = 64
OVERLAP = 32  # Historical name: this is the sliding step, not overlap length.
GAP = 20
NUM_JOINTS = 32
NUM_COORDS = 3


def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    min_samples = 3 * (order + 1) + 1
    if len(data) < min_samples:
        return data
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, data, axis=0)


def csv_loader(path):
    try:
        df = pd.read_csv(path, header=0)
        if df.shape[1] == 0:
            return None
        df = df.dropna()
        df = df.bfill()
        df = df.reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception as e:
        warnings.warn(f"Skipping unreadable CSV {path}: {e}", RuntimeWarning)
        return None


def compute_smv(acc_xyz):
    mean = np.mean(acc_xyz, axis=0, keepdims=True)
    zero_mean = acc_xyz - mean
    smv = np.sqrt(np.sum(zero_mean**2, axis=1, keepdims=True))
    return smv


def filter_repeated_ids(path):
    seen_a = set()
    seen_s = set()
    acc_ids = []
    skl_ids = []
    for a, s in path:
        if a not in seen_a and s not in seen_s:
            seen_a.add(a)
            seen_s.add(s)
            acc_ids.append(a)
            skl_ids.append(s)
    return (acc_ids, skl_ids)


def align_dtw(acc_filtered, acc_norm, skl, gap_mask):
    wrist = skl[:, 9, :]
    acc_frob = np.linalg.norm(acc_filtered, axis=1)
    wrist_frob = np.linalg.norm(wrist, axis=1)
    _, path = fastdtw(acc_frob[:, np.newaxis], wrist_frob[:, np.newaxis], dist=euclidean)
    acc_ids, skl_ids = filter_repeated_ids(path)
    return (acc_norm[acc_ids], skl[skl_ids], gap_mask[acc_ids])


def load_trial(acc_path, skl_path):
    acc_df = csv_loader(acc_path)
    if acc_df is None:
        return (None, None)
    acc_raw = acc_df.iloc[2:, -3:].to_numpy(dtype=np.float32)
    if len(acc_raw) < WINDOW:
        return (None, None)
    skl_df = csv_loader(skl_path)
    if skl_df is None:
        return (None, None)
    skl_raw = skl_df.iloc[:, -96:].to_numpy(dtype=np.float32)
    if skl_raw.shape[1] != 96:
        return (None, None)
    skl_raw = skl_raw.reshape(-1, NUM_JOINTS, NUM_COORDS)
    if len(skl_raw) < WINDOW:
        return (None, None)
    return (acc_raw, skl_raw)


def merge_trials(trial_list):
    trial_list = sorted(trial_list, key=lambda x: x[0])
    acc_parts = []
    skl_parts = []
    gap_parts = []
    acc_gap = np.zeros((GAP, 3), dtype=np.float32)
    skl_gap = np.zeros((GAP, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    for i, (_, acc_raw, skl_raw) in enumerate(trial_list):
        acc_parts.append(acc_raw)
        skl_parts.append(skl_raw)
        gap_parts.append(np.zeros(len(acc_raw), dtype=bool))
        if i < len(trial_list) - 1:
            acc_parts.append(acc_gap)
            skl_parts.append(skl_gap)
            gap_parts.append(np.ones(GAP, dtype=bool))
    merged_acc = np.concatenate(acc_parts)
    merged_skl = np.concatenate(skl_parts)
    gap_mask = np.concatenate(gap_parts)
    return (merged_acc, merged_skl, gap_mask)


def extract_windows_with_gap_mask(acc, skl, label, gap_mask):
    windows = []
    seen_ranges = set()
    magnitude = acc[:, 3]
    height = 1.4 if label == 1 else 1.2
    distance = 50 if label == 1 else 100
    peaks, _ = find_peaks(magnitude, height=height, distance=distance)

    def valid_window(start, end):
        if start < 0 or end > len(acc):
            return False
        if gap_mask[start:end].any():
            return False
        if (start, end) in seen_ranges:
            return False
        return True

    for p in peaks:
        start = max(0, p - WINDOW // 2)
        end = start + WINDOW
        if end > len(acc):
            end = len(acc)
            start = end - WINDOW
        if not valid_window(start, end):
            continue
        seen_ranges.add((start, end))
        windows.append((acc[start:end], skl[start:end]))
    for start in range(0, len(acc) - WINDOW + 1, OVERLAP):
        end = start + WINDOW
        if not valid_window(start, end):
            continue
        seen_ranges.add((start, end))
        windows.append((acc[start:end], skl[start:end]))
    return windows


def process_group(trial_list, label):
    if not trial_list:
        return []
    merged_acc, merged_skl, gap_mask = merge_trials(trial_list)
    if len(merged_acc) < WINDOW:
        return []
    acc_filtered = butterworth_filter(merged_acc)
    smv = compute_smv(acc_filtered)
    acc_with_smv = np.concatenate([acc_filtered, smv], axis=1)
    acc_norm = (acc_with_smv - acc_with_smv.mean(axis=0)) / (acc_with_smv.std(axis=0) + 1e-06)
    merged_skl = merged_skl - merged_skl[:, 0:1, :]
    acc_norm, merged_skl, gap_mask = align_dtw(acc_filtered, acc_norm, merged_skl, gap_mask)
    length = min(len(acc_norm), len(merged_skl), len(gap_mask))
    acc_norm = acc_norm[:length]
    merged_skl = merged_skl[:length]
    gap_mask = gap_mask[:length]
    if length < WINDOW:
        return []
    return extract_windows_with_gap_mask(acc_norm, merged_skl, label, gap_mask)


def build_dataset(data_root):
    acc_set = []
    skl_set = []
    labels = []
    subjects = []
    activities = []
    skeleton_files = glob.glob(f"{data_root}/young/skeleton/*.csv") + glob.glob(
        f"{data_root}/old/skeleton/*.csv"
    )
    print(f"\nSkeleton Files Found: {len(skeleton_files)}")
    group_map = defaultdict(list)
    for skel_file in skeleton_files:
        filename = os.path.basename(skel_file)
        match = re.findall("S\\d+A\\d+T\\d+", filename)
        if not match:
            continue
        desc = match[0]
        subject = int(re.search("S(\\d+)", desc).group(1))
        act_num = int(re.search("A(\\d+)", desc).group(1))
        trial_num = int(re.search("T(\\d+)", desc).group(1))
        group = "young" if "young" in skel_file else "old"
        acc_file = f"{data_root}/{group}/accelerometer/watch/{desc}.csv"
        if not os.path.exists(acc_file):
            continue
        group_map[subject, act_num].append((trial_num, acc_file, skel_file))
    print(f"\nUnique Groups: {len(group_map)}")
    total_windows = 0
    valid_groups = 0
    for (subject, act_num), file_entries in sorted(group_map.items()):
        label = int(act_num > 9)
        trial_data = []
        for trial_num, acc_path, skl_path in file_entries:
            acc_raw, skl_raw = load_trial(acc_path, skl_path)
            if acc_raw is None:
                continue
            trial_data.append((trial_num, acc_raw, skl_raw))
        if not trial_data:
            continue
        samples = process_group(trial_data, label)
        if not samples:
            continue
        valid_groups += 1
        total_windows += len(samples)
        for acc_w, skl_w in samples:
            acc_set.append(acc_w)
            skl_set.append(skl_w)
            labels.append(label)
            subjects.append(subject)
            activities.append(act_num)
    acc_set = np.array(acc_set, dtype=np.float32)
    skl_set = np.array(skl_set, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    subjects = np.array(subjects, dtype=np.int32)
    activities = np.array(activities, dtype=np.int32)
    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)
    print(f"Valid Groups      : {valid_groups}")
    print(f"Total Windows     : {total_windows}")
    print(f"Accelerometer     : {acc_set.shape}")
    print(f"Skeleton          : {skl_set.shape}")
    print(f"Labels            : {labels.shape}")
    print(f"Subjects          : {subjects.shape}")
    print(f"Activities        : {activities.shape}")
    print("=" * 60)
    return {
        "accelerometer": acc_set,
        "skeleton": skl_set,
        "labels": labels,
        "subjects": subjects,
        "activities": activities,
    }


def preprocess(data_root: str | Path, output: str | Path) -> dict:
    """Create the historical five-array NPZ from an explicitly supplied root."""
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    dataset = build_dataset(str(data_root))
    if len(dataset["labels"]) == 0:
        raise ValueError(
            "No valid windows. Expected young/old skeleton and accelerometer/watch CSV folders."
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **dataset)
    return {k: list(v.shape) for k, v in dataset.items()}
