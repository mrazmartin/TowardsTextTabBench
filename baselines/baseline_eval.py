import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

from tabpfn_client import TabPFNRegressor, TabPFNClassifier
from autogluon.tabular import TabularPredictor
from dataloader_functions.utils.df_downsample import downsample_rows_wrapper
from scipy.special import softmax

def _get_model_score(y_true, y_pred, proba, task_type):
    """
    Calculate the model score based on the task type.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - task_type: Type of task ('reg' for regression, 'clf' for classification)
    
    Returns:
    - score: Calculated score (R2 for regression, accuracy for classification)
    """
    if task_type == "reg":
        return r2_score(y_true, y_pred), root_mean_squared_error(y_true, y_pred)
    else:
        return accuracy_score(y_true, y_pred), log_loss(y_true, proba)

def evaluate_baseline(
    df: pd.DataFrame,
    df_name: str,
    label_col: str,
    task_type: str,
    textual_cols: list,
    seed: int = 42,
    k_folds: int = 5,
    max_samples: int = 10000,
        model: str = "TabPFNv2API",
    output_path: str = "/mnt/data/text_impact_results_5folds.csv"
):
    all_results = []
    # df_seed = df.sample(min(len(df), max_samples), random_state=seed).reset_index(drop=True)
    df_dict = downsample_rows_wrapper(
        {
            df_name: df,
        }, 
        label_col, 
        task_type, 
        max_samples,
        'stratified', 
        seed
    ) 
    df_seed = df_dict[df_name]
    print(f"Downsampled {len(df_seed)} rows for {df_name} dataset.")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(df_seed)):
        for use_text in [True, False]:
            suffix = "with_text" if use_text else "without_text"
            df_fold = df_seed.copy()

            if not use_text:
                df_fold = df_fold.drop(columns=textual_cols)

            train_df = df_fold.iloc[train_idx]
            test_df = df_fold.iloc[test_idx]

            y_test = test_df[label_col]
            X_test = test_df.drop(columns=[label_col])

            if model == "TabPFNv2API":
                # --- TabPFN v2 API---
                X_train_tabpfn = train_df.drop(columns=[label_col]).to_numpy()
                y_train_tabpfn = train_df[label_col].to_numpy()
                X_test_tabpfn = X_test.to_numpy()

                if task_type == "reg":
                    tabpfn_model = TabPFNRegressor()
                else:
                    tabpfn_model = TabPFNClassifier()
                    
                st = time.time()
                tabpfn_model.fit(X_train_tabpfn, y_train_tabpfn)
                tabpfn_preds = tabpfn_model.predict(X_test_tabpfn)
                fpt = time.time() - st

                proba = None
                if task_type == "clf":
                    proba = tabpfn_model.predict_proba(X_test_tabpfn)
                    # Ensure it's a NumPy array
                    if isinstance(proba, pd.DataFrame):
                        proba = proba.to_numpy()

                    # Ensure it's 2D
                    if proba.ndim == 1:
                        # Assume binary classification, convert to two-class
                        proba = np.vstack([1 - proba, proba]).T
                    # Normalize if rows don't sum to 1

                    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-2):
                        proba = softmax(proba, axis=1)

                    # Clip probabilities to avoid log(0) in log_loss 
                    proba = np.clip(proba, 1e-9, 1.0)
                    proba = proba / proba.sum(axis=1, keepdims=True)

                    if len(np.unique(y_test)) == 2:  # binary classification
                        roc_auc = roc_auc_score(y_test, proba[:, 1])
                    else:  # multiclass classification
                        roc_auc = roc_auc_score(y_test, proba, multi_class='ovr')
            
                tabpfn_score, loss = _get_model_score(y_test, tabpfn_preds, proba, task_type)

                all_results.append({
                    "fold": fold + 1,
                    "model": f"TabPFN_v2API_{suffix}",
                    "score": tabpfn_score,
                    "loss": loss,
                    "roc_auc": roc_auc if task_type == "clf" else None,
                    "fit_predict_time": fpt
                })

            elif model == "AGTabular":
                # --- AutoGluon Tabular ---
                agts = time.time()
                predictor = TabularPredictor(
                    label=label_col, 
                    verbosity=2, 
                    problem_type="regression" if task_type == "reg" else "binary", #replace to multiclass for multiclass classification
                    eval_metric="r2" if task_type == "reg" else "accuracy", 
                ).fit(
                    train_data=train_df, 
                    presets="best_quality",
                    time_limit=360,  # 1 hour)
                    dynamic_stacking=False,
                    ag_args_fit = {
                        "num_gpus": 1,
                        "num_cpus": 8,
                    }
                )
                ag_preds = predictor.predict(X_test)
                fpt_ag = time.time() - agts

                proba = None
                if task_type == "clf":
                    proba = predictor.predict_proba(X_test)
                    if isinstance(proba, pd.DataFrame):
                        proba = proba.to_numpy()
                    # Normalize if rows don't sum to 1
                    if proba.ndim == 2 and not np.allclose(proba.sum(axis=1), 1.0, atol=1e-2):
                        proba = softmax(proba, axis=1)

                    if len(np.unique(y_test)) == 2:  # binary classification
                        roc_auc = roc_auc_score(y_test, proba[:, 1])
                    else:  # multiclass classification
                        roc_auc = roc_auc_score(y_test, proba, multi_class='ovr')

                ag_score, ag_loss = _get_model_score(y_test, ag_preds, proba, task_type)

                all_results.append({
                    "fold": fold + 1,
                    "model": f"AutoGluon_Tabular_{suffix}",
                    "score": ag_score,
                    "loss": ag_loss,
                    "roc_auc": roc_auc if task_type == "clf" else None,
                    "fit_predict_time": fpt_ag
                })

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df
    
def plot_model_performance_summary(
    name: str,
    task: str,
    df: pd.DataFrame,
    metrics: list = ["score", "loss", "roc_auc"],
    base_dir: str = "../../baseline_results/plots"
) -> dict:
    """
    Plot mean ± std bar charts for given metrics and save in task-specific directories.

    Parameters:
    - name: Name of the dataset (used in plot title and file name)
    - task: 'reg' (regression) or 'clf' (classification)
    - df: DataFrame with columns like ['model', 'score', 'loss']
    - metrics: list of metric names to plot
    - base_dir: base path to save plots under

    Returns:
    - Dictionary of summary DataFrames per metric
    """
    summaries = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️ Skipping '{metric}' — not found in DataFrame.")
            continue

        summary = df.groupby("model")[metric].agg(['mean', 'std']).sort_values('mean', ascending=True)
        summaries[metric] = summary

        # Task-specific metric labeling
        if metric == "score":
            if task == "reg":
                metric_label = "R² Score"
            else:
                metric_label = "Accuracy Score"
        elif metric == "loss":
            if task == "reg":
                metric_label = "RMSE"
            else:
                metric_label = "Log Loss"
        else:
            metric_label = metric.capitalize()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(
            summary.index,
            summary['mean'],
            xerr=summary['std'],
            capsize=5,
            color='skyblue',
            edgecolor='black'
        )
        plt.xlabel(f"Mean ± Std of {metric_label}")
        plt.ylabel("Model and Usage of Textual Features")
        plt.title(f"Model {metric_label} Comparison for {name} Dataset")
        plt.grid(axis='x')
        plt.tight_layout()

        # Save
        save_path = os.path.join(base_dir, task, metric)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to {save_path}")
        save_path = os.path.join(save_path, f"{name}.png")
        plt.savefig(save_path)
        plt.close()

    return summaries