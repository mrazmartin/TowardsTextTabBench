import numpy as np
import pandas as pd

import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets_notebooks.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg, color_text

def downsample_rows_stratified(df_dict, target_col, task, downsampled_rows, seed=0):
    """
    Downsamples each DataFrame in df_dict to 'downsampled_rows' total rows.
    - For regression: random sampling.
    - For classification: stratified with class balancing logic.
    - Returns a dict of downsampled DataFrames.
    """

    out_dict = {}

    for df_name, df in df_dict.items():
        info_msg(f"\n[INFO] Downsampling dataframe: {df_name} (original rows: {len(df)})")

        np.random.seed(seed)  # reset seed per df for reproducibility

        if task == 'reg':
            if len(df) <= downsampled_rows:
                info_msg(f"Dataset has only {len(df)} rows. No downsampling needed.", color='yellow')
                out_dict[df_name] = df.copy()
                continue

            out_dict[df_name] = df.sample(n=downsampled_rows, random_state=seed).reset_index(drop=True)
            continue

        elif task == 'clf':
            unique_classes = df[target_col].unique()
            n_classes = len(unique_classes)

            target_per_class = downsampled_rows // n_classes
            info_msg(f"Trying to sample ~{target_per_class} rows per class (total={downsampled_rows})", color='yellow')

            selected_indices = []
            remaining_pool = []

            for cls in unique_classes:
                cls_indices = df[df[target_col] == cls].index.to_numpy()

                if len(cls_indices) <= target_per_class:
                    info_msg(f"[INFO] Class {cls} has only {len(cls_indices)} instances. Keeping all.", color='green')
                    selected_indices.extend(cls_indices)
                else:
                    sampled_cls_indices = np.random.choice(cls_indices, target_per_class, replace=False)
                    selected_indices.extend(sampled_cls_indices)

                    remaining_pool.extend(list(set(cls_indices) - set(sampled_cls_indices)))

            n_missing = downsampled_rows - len(selected_indices)

            if n_missing > 0 and remaining_pool:
                if n_missing >= len(remaining_pool):
                    info_msg(f"Filling up with all remaining {len(remaining_pool)} samples.", color='green')
                    selected_indices.extend(remaining_pool)
                else:
                    extra_indices = np.random.choice(remaining_pool, n_missing, replace=False)
                    selected_indices.extend(extra_indices)

            selected_df = df.loc[selected_indices].sample(frac=1, random_state=seed).reset_index(drop=True)

            # Per-class counts after downsampling
            class_counts = selected_df[target_col].value_counts().to_dict()
            class_count_str = ', '.join([f"{cls}: {count}" for cls, count in class_counts.items()])

            info_msg(f"Final downsampled dataset has {len(selected_df)} rows. Per class counts: [{class_count_str}]\n", color='cyan')

            out_dict[df_name] = selected_df

        else:
            raise ValueError(f"Unknown task type: {task}. Should be 'reg' or 'clf'.")

    return out_dict

def downsample_rows_random(df_dict, downsampled_rows, seed=0):
    """
    Purely random downsampling of each DataFrame in df_dict to 'downsampled_rows' total rows.
    - Ignores task type, class balance, etc.
    - Returns a dict of downsampled DataFrames.
    """

    out_dict = {}

    for df_name, df in df_dict.items():
        info_msg(f"\n[INFO] Randomly downsampling dataframe: {df_name} (original rows: {len(df)})")

        np.random.seed(seed)  # reset seed per df for reproducibility

        if len(df) <= downsampled_rows:
            info_msg(f"[INFO] Dataset has only {len(df)} rows. No downsampling needed.", color='yellow')
            out_dict[df_name] = df.copy()
            continue

        downsampled_df = df.sample(n=downsampled_rows, random_state=seed).reset_index(drop=True)
        info_msg(f"[INFO] Downsampled to {len(downsampled_df)} rows.", color='cyan')

        out_dict[df_name] = downsampled_df

    return out_dict


def downsample_rows_wrapper(df_dict, target_col, task, downsampled_rows, mode='stratified', seed=0):
    """
    Wrapper to downsample rows of each DataFrame in df_dict.
    - mode: 'stratified' → uses stratified per-class logic (only for classification)
            'random' → purely random downsampling
    - task: 'reg' or 'clf' (only relevant for stratified)
    """

    if mode == 'random':
        print(f"\n[WRAPPER] Running random downsampling mode.")
        return downsample_rows_random(df_dict, downsampled_rows, seed=seed)

    elif mode == 'stratified':
        if task not in ['reg', 'clf']:
            raise ValueError(f"[WRAPPER] Invalid task '{task}' for stratified mode. Should be 'reg' or 'clf'.")
        print(f"\n[WRAPPER] Running stratified downsampling mode for task: {task}")
        return downsample_rows_stratified(df_dict, target_col, task, downsampled_rows, seed=seed)

    else:
        raise ValueError(f"[WRAPPER] Unknown mode: '{mode}'. Choose 'random' or 'stratified'.")
