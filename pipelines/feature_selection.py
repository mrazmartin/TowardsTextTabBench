import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.metrics import accuracy_score
from datasets_notebooks.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg, color_text

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from pipelines.embedd_text import load_embedded_dataset
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import shap
import signal, time
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

def safe_impute(X):
    """
    Imputes NaNs in DataFrame or ndarray using per-column heuristics:
    - Numeric-like → mean
    - Categorical-like (object dtype or low cardinality) → mode
    """
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    for col in X_df.columns:
        if X_df[col].isnull().any():
            n_unique = X_df[col].nunique(dropna=True)
            if X_df[col].dtype == 'object' or n_unique < 30:
                fill_value = X_df[col].mode(dropna=True).iloc[0]
            else:
                fill_value = X_df[col].mean(skipna=True)

            X_df[col] = X_df[col].fillna(fill_value)

    return X_df

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

#### downsampling functions ####

def t_test_feature_selection(X, y, alpha=0.05, top_k=50):
    """
    Performs T-test feature selection for binary classification.
    Returns indices of top_k most significant features after Bonferroni correction.
    """

    # Ensure X is DataFrame for column convenience
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    # Ensure y is numpy array
    y_arr = np.asarray(y)

    # Check binary target
    classes = np.unique(y_arr)
    if len(classes) != 2:
        raise ValueError(f"T-test requires exactly 2 classes. Found {len(classes)}: {classes}")

    # Split data by classes
    mask_class0 = y_arr == classes[0]
    mask_class1 = y_arr == classes[1]

    # Perform independent two-sample t-test per feature
    t_vals, p_vals = ttest_ind(X_df[mask_class0], X_df[mask_class1], axis=0, equal_var=False, nan_policy='omit')

    # Bonferroni correction
    corrected_alpha = alpha / X_df.shape[1]

    # Select significant features
    significant_dims = np.where(p_vals < corrected_alpha)[0]

    if len(significant_dims) == 0:
        print(f"[T-test] No significant features found (Bonferroni corrected alpha={corrected_alpha}). Returning empty.")
        return significant_dims  # empty array

    # Sort significant features by p-value ascending (most significant first)
    sorted_significant_dims = significant_dims[np.argsort(p_vals[significant_dims])]

    # Limit to top_k
    top_k = min(top_k, len(sorted_significant_dims))
    selected_dims = sorted_significant_dims[:top_k]

    # print(f"[T-test] Selected top {top_k} significant features: {selected_dims}")
    return selected_dims

def anova_feature_selection(X, y, alpha=0.05, top_k=50):
    """
    Performs ANOVA F-test feature selection for multi-class classification.
    Returns indices of top_k most significant features (after Bonferroni correction and ranking by p-value).
    """

    # Convert X to DataFrame for convenience
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    # Ensure y is numpy array
    y_arr = np.asarray(y)

    # Unique class labels
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        raise ValueError(f"ANOVA requires at least 2 classes. Found {len(unique_classes)}")

    # Group data by class
    data_by_class = [X_df[y_arr == cls] for cls in unique_classes]

    # Compute ANOVA p-values for each feature
    anova_p_values = []
    for col in X_df.columns:
        groups = [group[col] for group in data_by_class]
        stat, p_val = f_oneway(*groups)
        anova_p_values.append(p_val)

    anova_p_values = np.array(anova_p_values)

    # Bonferroni correction threshold
    corrected_alpha = alpha / X_df.shape[1]

    # Find significant features (p < corrected_alpha)
    significant_dims = np.where(anova_p_values < corrected_alpha)[0]

    if len(significant_dims) == 0:
        print(f"[ANOVA] No significant features found (Bonferroni corrected alpha={corrected_alpha}). Returning empty.")
        return significant_dims  # empty array

    # Sort significant features by p-value (ascending = more significant)
    sorted_significant_dims = significant_dims[np.argsort(anova_p_values[significant_dims])]

    # Limit to top_k
    top_k = min(top_k, len(sorted_significant_dims))
    selected_dims = sorted_significant_dims[:top_k]

    # print(f"[ANOVA] Selected top {top_k} significant features: {selected_dims}")
    return selected_dims

def variance_based_selection(X_train, top_k=50):
    """
    Selects indices of top-k features with highest variance.
    Unsupervised filter (task-agnostic).
    """

    # Compute variance for each feature (axis=0 is per column)
    variances = np.var(X_train, axis=0)

    # Safety cap: if top_k > num_features
    top_k = min(top_k, X_train.shape[1])

    # Get indices of top-k variances in descending order
    top_k_dims = np.argsort(variances)[::-1][:top_k]

    # print(f"Top {top_k} features by variance: {top_k_dims}")
    return top_k_dims

def pca_feature_selection(X, n_keep=50, seed=0):
    """
    Selects indices of most important features based on PCA loadings weighted by explained variance.
    """

    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        X_fixed = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    else:
        X_fixed = X.copy()

    # NaN handling heuristics
    for col in X_fixed.columns:
        if X_fixed[col].isnull().any():
            n_unique = X_fixed[col].nunique(dropna=True)
            if X_fixed[col].dtype == 'object' or n_unique < 30:
                fill_value = X_fixed[col].mode(dropna=True).iloc[0]
            else:
                fill_value = X_fixed[col].mean(skipna=True)
            X_fixed[col] = X_fixed[col].fillna(fill_value)

    # Run PCA without scaling (raw variances matter)
    n_components = min(X_fixed.shape[0], X_fixed.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_fixed)

    # Absolute loadings
    loadings = np.abs(pca.components_)  # shape: (n_components, n_features)

    # Weight loadings by explained variance ratio
    weighted_loadings = loadings * pca.explained_variance_ratio_[:, np.newaxis]

    # Aggregate importance of each feature
    feature_importance = weighted_loadings.sum(axis=0)

    # Rank features by importance
    top_feature_indices = np.argsort(feature_importance)[::-1][:n_keep]

    # print(f"Selected feature indices (top {n_keep} by PCA importance): {top_feature_indices}")
    return top_feature_indices

def l1_feature_selection(X, y, task='classification', alpha=0.01, seed=0, top_k=50):
    """
    L1-based feature selection.
    - task: 'regression' or 'classification'.
    - Uses Lasso for regression.
    - Uses LogisticRegression with L1 penalty for classification.
    - Returns indices of top_k most important features (by absolute coefficient).
    """

    # Impute NaNs
    X_df = safe_impute(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Train model with L1 penalty
    if task == 'reg':
        print(f"Task: Regression → Lasso (alpha={alpha})")
        model = Lasso(alpha=alpha, random_state=seed, max_iter=10000)
        model.fit(X_scaled, y)
        coef = model.coef_
    elif task == 'clf':
        print(f"Task: Classification → LogisticRegression L1 (alpha={alpha})")
        model = LogisticRegression(penalty='l1', solver='saga', C=1/alpha, random_state=seed, max_iter=10000)
        model.fit(X_scaled, y)
        coef = model.coef_[0]  # shape: (n_features,)
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")

    # Get non-zero coefficients
    non_zero_indices = np.where(coef != 0)[0]

    if len(non_zero_indices) == 0:
        print("[L1] No features selected with non-zero coefficients. Returning empty selection.")
        return non_zero_indices  # empty array

    # Rank by absolute coefficient magnitude
    abs_coef = np.abs(coef[non_zero_indices])
    top_k = min(top_k, len(non_zero_indices))
    top_indices = non_zero_indices[np.argsort(abs_coef)[::-1][:top_k]]

    # print(f"[L1] Selected top {top_k} features: {top_indices}")
    return top_indices

def correlation_feature_selection(X, y, method='pearson', top_k=50, task=None):
    """
    Selects top-k features based on Pearson/Spearman correlation with the target.
    Works for regression and binary classification tasks.
    """

    # Convert X to DataFrame if needed
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    y_arr = np.asarray(y)

    # Guard: Block multi-class classification only
    if task == 'clf' and len(np.unique(y_arr)) > 2:
        raise ValueError(f"Correlation feature selection is not recommended for multi-class classification. Use ANOVA instead.")

    # Compute correlations
    correlations = []
    for col in X_df.columns:
        if method == 'pearson':
            corr = np.corrcoef(X_df[col], y_arr)[0, 1]
        elif method == 'spearman':
            corr, _ = spearmanr(X_df[col], y_arr)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        correlations.append(corr)

    correlations = np.array(correlations)

    # Rank features by absolute correlation
    important_dims = np.argsort(np.abs(correlations))[::-1][:top_k]

    # print(f"Top {top_k} correlated features (method={method}): {important_dims}")
    return important_dims

def shap_feature_selection(X, y, task='clf', top_k=50, model_type='xgb', seed=0, time_limit_sec=60):
    """
    Selects top-k important features based on SHAP values.
    - task: 'classification' or 'regression'.
    - model_type: 'xgb' (recommended) or 'linear' for LogisticRegression/LinearRegression.
    - Returns indices of selected features ranked by global importance.
    """
    # Convert X to DataFrame if needed
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    # Train model
    if model_type == 'xgb':
        model = XGBClassifier(random_state=seed, verbosity=0) if task == 'classification' else XGBRegressor(random_state=seed, verbosity=0)
    elif model_type == 'linear':
        model = LogisticRegression(penalty='l1', solver='saga', random_state=seed, max_iter=10000) if task == 'classification' else LinearRegression()
    else:
        raise ValueError("model_type must be 'xgb' or 'linear'")

    model.fit(X_df, y)

    # Setup timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(time_limit_sec)

    try:
        print(f"[SHAP] Computing SHAP values with time limit {time_limit_sec}s ...")
        explainer = shap.Explainer(model, X_df)
        shap_values = explainer(X_df)
        signal.alarm(0)  # Cancel the alarm if finished in time
    except TimeoutException:
        print(f"[SHAP WARNING] SHAP computation exceeded time limit ({time_limit_sec}s). Skipping SHAP-based selection.")
        # Fallback: random selection or top variance features
        variances = np.var(X_df.values, axis=0)
        top_k = min(top_k, X_df.shape[1])
        top_feature_indices = np.argsort(variances)[::-1][:top_k]
        print(f"[SHAP Fallback] Selected top-{top_k} features by variance instead.")
        return top_feature_indices

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Rank features by importance
    top_k = min(top_k, X_df.shape[1])
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    # print(f"[SHAP] Selected top-{top_k} features by SHAP importance: {top_feature_indices}")
    return top_feature_indices

##### TabPFN evaluation #####

def test_on_tabpfn_v2(X_full, y_full):
    # Downsample to 1000 rows (stratified to preserve class distribution)
    if len(y_full) > 1000:
        X_downsampled, _, y_downsampled, _ = train_test_split(
            X_full, y_full, train_size=1000, random_state=42, stratify=y_full
        )
    else:
        X_downsampled = X_full
        y_downsampled = y_full
        print(f"Dataset has less than 1000 samples ({len(y_full)}), using all.")

    # Train/test split on downsampled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_downsampled, y_downsampled, test_size=0.2, random_state=42, stratify=y_downsampled
    )

    # Initialize and train TabPFNClassifier
    tabpfn_model = TabPFNClassifier(device="cpu", seed=0)
    tabpfn_model.fit(X_train, y_train)

    # Predict and evaluate
    pred = tabpfn_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("Accuracy score: ", acc)

    return acc

#### Main downsampling function ####

def is_selector_applicable(strategy, task, y):
    """
    Checks whether a feature selection strategy is applicable for the given task & target.
    Returns True (run) or False (skip).

    strategy: str, feature selection method (e.g., 't-test', 'anova', 'shap', ...)
    task: str, 'reg' or 'clf'
    y: target vector (numpy array or pd.Series)
    """

    # Determine task type
    if task == 'reg':
        task_type = 'regression'
    elif task == 'clf':
        n_classes = len(np.unique(y))
        task_type = 'binary' if n_classes == 2 else 'multiclass'
    else:
        raise ValueError(f"Unknown task type: {task}. Must be 'reg' or 'clf'.")

    # Mapping of selectors to allowed task types
    strategy_task_mapping = {
        't-test': ['binary'],                  # Only binary classification
        'anova': ['binary', 'multiclass'],     # Works for all classification
        'lasso': ['binary', 'multiclass', 'regression'],
        'variance': ['binary', 'multiclass', 'regression'],  # unsupervised, always fine
        'pca': ['binary', 'multiclass', 'regression'],       # unsupervised, always fine
        'correlation': ['regression'],          # Only regression tasks
        'shap': ['binary', 'multiclass', 'regression'],
        'random': ['binary', 'multiclass', 'regression'],    # always fine
    }

    allowed_types = strategy_task_mapping.get(strategy, None)

    if allowed_types is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    if task_type not in allowed_types:
        info_msg(f"Skipping '{strategy}' for task '{task_type}' — not applicable.", color='yellow')
        return False

    return True

def downsample_features(emd_df_dic, config_df, summary_df, max_features=500, strat_list=['t-test', 'anova', 'lasso', 'variance', 'pca', 'correlation', 'shap', 'random'], seed=0):
    """
    For each embedding dataframe, downsample the features using the selected strategy.

    Always keep the default columns (categorical and numerical) and the target column.
    The rest of the columns are downsampled according to the selected strategy.

    When the maximum number of features is not reached via supervised strategies, the rest of the columns are picked randomly.

    Returns a dictionary with the downsampled dataframes with the original column names.
    The keys are the names of the embedding dataframes + the strategy name.
    """
    # Extract target column name and task type
    target_col = config_df['target']
    task = config_df['task']

    # Set seed for random
    np.random.seed(seed)

    # Default columns to keep (categorical + numerical)
    default_cols_names = summary_df.loc[summary_df['Type'].isin(['categorical', 'numerical']), 'Column Name'].tolist()
    print(f"Default columns to keep: {default_cols_names}")

    out_dict = {}

    for emd_df_key, emb_df in emd_df_dic.items():

        # Identify embedding columns (candidates for downsampling)
        embedding_cols = [col for col in emb_df.columns if col not in default_cols_names + [target_col]]

        # Sanity check for allowed features
        max_features_per_emd = max_features - len(default_cols_names)

        # check if we dont have all features just in default columns
        if max_features_per_emd <= 0:
            warn_msg(f"[WARN] max_features ({max_features}) too small. Using only default columns.")
            if target_col not in default_cols_names:
                out_dict[emd_df_key + "_default"] = emb_df[default_cols_names + [target_col]]
            else:
                out_dict[emd_df_key + "_default"] = emb_df[default_cols_names]
            continue

        # if there are less features than max_features_per_emd, use all of them
        if len(embedding_cols) <= max_features_per_emd:
            info_msg(f"Downsampling less features ({len(embedding_cols)}) than is the max_features_per_emd ({max_features_per_emd}) -> Using all features.", color='green')
            selected_features = np.arange(len(embedding_cols))
            selected_cols_names = embedding_cols

            if target_col not in default_cols_names:
                selected_cols_names.append(target_col)

            out_dict[emd_df_key + "_all"] = emb_df[default_cols_names + selected_cols_names]
            continue

        info_msg(f"Features per embedding: {max_features_per_emd} (out of {max_features})", color='green')

        # Prepare X and y
        X_full = emb_df[embedding_cols].values
        y_full = emb_df[target_col].values

        for strat in strat_list:
            # Check applicability before selection
            if not is_selector_applicable(strat, task, y_full):
                continue

            out_name = emd_df_key + "_" + strat

            if strat == 't-test':
                selected_features = t_test_feature_selection(X_full, y_full, alpha=0.05, top_k=max_features_per_emd)
            elif strat == 'anova':
                selected_features = anova_feature_selection(X_full, y_full, alpha=0.05, top_k=max_features_per_emd)
            elif strat == 'lasso':
                selected_features = l1_feature_selection(X_full, y_full, task=task, alpha=0.01, seed=seed, top_k=max_features_per_emd)
            elif strat == 'variance':
                selected_features = variance_based_selection(X_full, top_k=max_features_per_emd)
            elif strat == 'pca':
                selected_features = pca_feature_selection(X_full, n_keep=max_features_per_emd, seed=seed)
            elif strat == 'correlation':
                selected_features = correlation_feature_selection(X_full, y_full, method='pearson', top_k=max_features_per_emd, task=task)
            elif strat == 'shap':
                selected_features = shap_feature_selection(X_full, y_full, task=task, top_k=max_features_per_emd, model_type='xgb', time_limit_sec=60, seed=seed)
            elif strat == 'random':
                if X_full.shape[1] >= max_features_per_emd:
                    selected_features = np.random.choice(X_full.shape[1], max_features_per_emd, replace=False)
                else:
                    selected_features = np.arange(X_full.shape[1])
                    print(f"Dataset has only {X_full.shape[1]} embedding features. Using all.")
            else:
                raise ValueError(f"Unknown strategy: {strat}")


            # if not full amount of features selected, pick the rest randomly
            if len(selected_features) < max_features_per_emd:
                info_msg(f"Selected {len(selected_features)} / {max_features_per_emd} features via '{strat}'. Picking the rest randomly.", color='orange')
                remaining_features = np.setdiff1d(np.arange(X_full.shape[1]), selected_features)
                additional_features = np.random.choice(remaining_features, max_features_per_emd - len(selected_features), replace=False)
                selected_features = np.concatenate((selected_features, additional_features))
            else:
                info_msg(f"Selected all {len(selected_features)} features using {strat} strategy.")

            # Get selected embedding columns by name
            selected_cols_names = [embedding_cols[i] for i in selected_features]

            # Final columns: default + selected + target
            final_cols = default_cols_names + selected_cols_names
            if target_col not in final_cols:
                final_cols.append(target_col)

            selected_df = emb_df[final_cols]
            out_dict[out_name] = selected_df

    return out_dict

###
# Testing downsample correctness
###
def generate_simple_test_data(num_rows=300, num_features=10, signal_shift=2.0, noise_std=0.1):
    """
    Generates test data where f0 has a clear mean difference between classes.
    Ensures T-test and PCA can pick it up.
    """
    np.random.seed(42)

    # Binary target assignment (balanced)
    target = np.random.choice(['class1', 'class0'], size=num_rows)

    # f0 has shifted mean for class1 vs class0
    f0 = np.random.randn(num_rows) * noise_std
    f0[target == 'class1'] += signal_shift  # Shift mean up for class1
    f0[target == 'class0'] -= signal_shift  # Shift mean down for class0

    # f1 weakly correlated with target
    f1 = np.random.rand(num_rows)

    # Other features: pure noise
    noise_features = {f"f{i}": np.random.rand(num_rows) for i in range(2, num_features)}

    # Assemble DataFrame
    df = pd.DataFrame({'f0': f0, 'f1': f1, **noise_features, 'target': target})

    # Config
    config_df = pd.Series({'target': 'target', 'task': 'classification'})

    # Summary (no default columns)
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    emd_df_dic = {'test_df': df}

    return emd_df_dic, config_df, summary_df

def generate_pca_sensitive_test_data(num_rows=300, num_features=10, f0_std=10.0, noise_std=0.1):
    """
    Generate data where f0 has dominant variance, so PCA picks it up.
    """

    np.random.seed(42)

    # f0 → large variance
    f0 = np.random.randn(num_rows) * f0_std

    # f1 → weak variance
    f1 = np.random.randn(num_rows) * noise_std

    # Other noise features → tiny variance
    noise_features = {f"f{i}": np.random.randn(num_rows) * noise_std for i in range(2, num_features)}

    # Binary target (for pipeline compatibility, though PCA ignores it)
    target = np.random.choice(['class1', 'class0'], size=num_rows)

    # Assemble DataFrame
    df = pd.DataFrame({'f0': f0, 'f1': f1, **noise_features, 'target': target})

    # Config
    config_df = pd.Series({'target': 'target', 'task': 'classification'})

    # No default columns
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    emd_df_dic = {'test_df': df}

    return emd_df_dic, config_df, summary_df

def generate_ttest_friendly_downsample_data(num_rows=300, num_features=10, signal_shift=3.0, noise_std=1.0):
    """
    Generates inputs for downsample_features() to test T-test feature selection.
    - f0 has large class-based mean difference.
    - Other features are pure noise.
    - Returns: emd_df_dic, config_df, summary_df.
    """

    np.random.seed(42)

    # Binary target assignment (balanced)
    y = np.random.choice(['class0', 'class1'], size=num_rows)

    # f0: signal feature with shifted means between classes
    f0 = np.random.randn(num_rows) * noise_std
    f0[y == 'class1'] += signal_shift
    f0[y == 'class0'] -= signal_shift

    # Remaining features: pure noise
    feature_data = {'f0': f0}
    for i in range(1, num_features):
        feature_data[f"f{i}"] = np.random.randn(num_rows) * noise_std

    # Target column
    feature_data['target'] = y

    # Embedding DataFrame dictionary
    emd_df_dic = {'test_df': pd.DataFrame(feature_data)}

    # Config DataFrame
    config_df = pd.Series({'target': 'target', 'task': 'classification'})

    # Summary DataFrame (no default columns in this test)
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    return emd_df_dic, config_df, summary_df

def generate_anova_friendly_downsample_data(num_rows_per_class=100, num_classes=3, num_features=10, signal_shift=3.0, noise_std=1.0):
    """
    Generates inputs for downsample_features() to test ANOVA feature selection.
    - f0 has distinct means across classes.
    - Other features are pure noise.
    - Returns: emd_df_dic, config_df, summary_df.
    """

    np.random.seed(42)

    # Target variable (multi-class)
    y = np.concatenate([[f"class{i}" for i in range(num_classes)] * num_rows_per_class])
    total_rows = len(y)

    # f0: signal feature with per-class mean shift
    f0 = np.zeros(total_rows)
    for i in range(num_classes):
        class_mask = np.where(y == f"class{i}")[0]
        f0[class_mask] = np.random.randn(len(class_mask)) * noise_std + (i * signal_shift)

    # Other features: pure noise
    feature_data = {'f0': f0}
    for j in range(1, num_features):
        feature_data[f"f{j}"] = np.random.randn(total_rows) * noise_std

    # Target column
    feature_data['target'] = y

    # Embedding DataFrame dict
    emd_df_dic = {'test_df': pd.DataFrame(feature_data)}

    # Config DataFrame (as Series)
    config_df = pd.Series({'target': 'target', 'task': 'classification'})

    # Summary DataFrame (no default columns in this test)
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    return emd_df_dic, config_df, summary_df

def generate_correlation_friendly_downsample_data(
    num_rows=300,
    num_features=10,
    task='reg',
    signal_feature='f0',
    signal_strength=10.0,  # stronger to dominate
    noise_std=1.0
):
    """
    Generates synthetic data for correlation-based feature selection.
    Strong signal injected into `signal_feature` (f0).
    """

    np.random.seed(42)  # <-- remove fixed seed, or use time-based randomness

    # Generate noise features
    feature_data = {f"f{i}": np.random.randn(num_rows) * noise_std for i in range(num_features)}

    # Inject strong signal into chosen feature (f0)
    signal_values = feature_data[signal_feature] * signal_strength

    # Target generation
    if task == 'reg':
        target = signal_values + np.random.randn(num_rows) * noise_std
    elif task == 'binary_classification':
        logits = signal_values + np.random.randn(num_rows) * noise_std
        probs = 1 / (1 + np.exp(-logits))
        target = (probs > 0.5).astype(int)
    else:
        raise ValueError("Task must be 'regression' or 'binary_classification'")

    # Final dataframe
    feature_data['target'] = target

    # Outputs
    emd_df_dic = {'test_df': pd.DataFrame(feature_data)}
    config_df = pd.Series({'target': 'target', 'task': task})
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    print(f"Injected signal into {signal_feature} with strength {signal_strength}")
    return emd_df_dic, config_df, summary_df

def generate_l1_friendly_data(
    num_rows=300,
    num_features=10,
    task='reg',  # or 'classification'
    signal_feature='f0',
    signal_strength=5.0,
    noise_std=1.0
):
    """
    Generates data where L1 (Lasso/LogReg) should select `signal_feature` as important.
    - Sparse signal in one feature (f0 by default).
    - Other features are irrelevant noise.
    """

    np.random.seed(42)

    # Generate noise for all features
    feature_data = {f"f{i}": np.random.randn(num_rows) * noise_std for i in range(num_features)}

    # Inject sparse signal into chosen feature (f0)
    signal = feature_data[signal_feature] * signal_strength

    # Generate target
    if task == 'reg':
        target = signal + np.random.randn(num_rows) * noise_std  # continuous target
    elif task == 'clf':
        logits = signal + np.random.randn(num_rows) * noise_std
        probs = 1 / (1 + np.exp(-logits))
        target = (probs > 0.5).astype(int)  # binary target
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    # Add target column
    feature_data['target'] = target

    # Return pipeline-friendly dicts
    emd_df_dic = {'test_df': pd.DataFrame(feature_data)}
    config_df = pd.Series({'target': 'target', 'task': task})
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})  # no default columns

    print(f"Injected L1-sparse signal into {signal_feature} with strength {signal_strength}")
    return emd_df_dic, config_df, summary_df

def generate_variance_friendly_data(
    num_rows=300,
    num_features=10,
    high_variance_feature='f0',
    high_variance_std=10.0,
    low_variance_std=1.0,
    task='regression'
):
    """
    Generates data where one feature (e.g. f0) has much higher variance than others.
    Suitable for testing variance-based feature selection.
    """

    np.random.seed(42)

    feature_data = {}

    for i in range(num_features):
        std = high_variance_std if f"f{i}" == high_variance_feature else low_variance_std
        feature_data[f"f{i}"] = np.random.randn(num_rows) * std

    # Create arbitrary target (not used by variance selector, but needed for pipeline)
    if task == 'reg':
        target = np.random.randn(num_rows)
    elif task == 'clf':
        target = np.random.randint(0, 2, size=num_rows)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    feature_data['target'] = target

    emd_df_dic = {'test_df': pd.DataFrame(feature_data)}
    config_df = pd.Series({'target': 'target', 'task': task})
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    print(f"Injected high variance into {high_variance_feature} (std={high_variance_std})")
    return emd_df_dic, config_df, summary_df

def generate_shap_friendly_data(
    num_rows=300,
    num_features=10,
    task='reg',  # or 'classification'
    signal_feature='f0',
    signal_strength=5.0,
    noise_std=1.0
):
    """
    Generates synthetic data where `signal_feature` has dominant influence on the target.
    Guarantees SHAP will rank it as top-1 important.
    """

    np.random.seed(42)

    # Generate random noise features
    feature_data = {f"f{i}": np.random.randn(num_rows) * noise_std for i in range(num_features)}

    # Inject signal into chosen feature (f0 by default)
    signal = feature_data[signal_feature] * signal_strength

    # Generate target based on signal feature
    if task == 'reg':
        target = signal + np.random.randn(num_rows) * noise_std
    elif task == 'clf':
        logits = signal + np.random.randn(num_rows) * noise_std
        probs = 1 / (1 + np.exp(-logits))
        target = (probs > 0.5).astype(int)
    else:
        raise ValueError("Task must be 'reg' or 'clf'")

    # Add target column
    feature_data['target'] = target

    # Return pipeline-friendly outputs
    emd_df_dic = {'test_df': pd.DataFrame(feature_data)}
    config_df = pd.Series({'target': 'target', 'task': task})
    summary_df = pd.DataFrame({'Column Name': [], 'Type': []})

    print(f"Injected SHAP-dominant signal into {signal_feature} with strength {signal_strength}")
    return emd_df_dic, config_df, summary_df

if __name__ == "__main__":

    if True:
        # emd_df_dic, config_df, summary_df = generate_pca_sensitive_test_data(num_rows=300, num_features=10, f0_std=10.0, noise_std=0.1)
        # emd_df_dic, config_df, summary_df = generate_ttest_friendly_downsample_data(num_rows=300, num_features=10, signal_shift=3.0)
        # emd_df_dic, config_df, summary_df = generate_anova_friendly_downsample_data(num_rows_per_class=100, num_classes=3, num_features=10, signal_shift=3.0)
        # emd_df_dic, config_df, summary_df = generate_correlation_friendly_downsample_data(num_rows=300, num_features=10, task='reg', signal_feature='f0', signal_strength=5.0)
        # emd_df_dic, config_df, summary_df = generate_l1_friendly_data(num_rows=300, num_features=10, task='clf', signal_feature='f0', signal_strength=5.0)
        # emd_df_dic, config_df, summary_df = generate_variance_friendly_data(num_rows=300, num_features=10, high_variance_feature='f0',
        #                                                                    high_variance_std=10.0, low_variance_std=1.0, task='reg')
        emd_df_dic, config_df, summary_df = generate_shap_friendly_data(num_rows=300, num_features=10, task='reg', signal_feature='f0', signal_strength=5.0)

        out_dict = downsample_features(emd_df_dic, config_df, summary_df, max_features=5, strat_list=['shap'])

        for name, df in out_dict.items():
            print(f"{name}: {df.columns.tolist()}")

    else:
        list_of_strategies = ['pca']

        dataset_name = "hs_cards"
        max_features = 50

        # Load embeddings & config
        emb_df, config_df, summary_df = load_embedded_dataset(dataset_name, methods=['skrub'], save_format='npy')

        print("The shape of the dataset: ", emb_df['skrub_df'].shape)
        print("Target column is:", config_df['target'])

        # Extract target column name
        target_col = config_df['target']

        # Prepare X and y
        y_full = emb_df['skrub_df'][target_col].values  # labels
        X_full = emb_df['skrub_df'].drop(target_col, axis=1).values  # features

        #  test_on_tabpfn_v2(X_full, y_full)

        for strat in list_of_strategies:

            print(f"Running feature selection with strategy: {strat}")

            if strat == 't-test':
                selected_features = t_test_feature_selection(X_full, y_full, alpha=0.05)
                X_ds = X_full[:, selected_features]

            elif strat == 'anova':
                selected_features = anova_feature_selection(X_full, y_full, alpha=0.05)
                X_ds = X_full[:, selected_features]

            elif strat == 'lasso':
                selected_features = l1_feature_selection(X_full, y_full, task='classification', alpha=0.01)
                X_ds = X_full[:, selected_features]

            elif strat == 'variance':
                selected_features = variance_based_selection(X_full, top_k=max_features)
                X_ds = X_full[:, selected_features]

            elif strat == 'pca':
                # X_ds = pca_dimensionality_reduction(X_full, n_components=100)
                selected_features = pca_feature_selection(X_full, n_keep=max_features)
                X_ds = X_full[:, selected_features]

            elif strat == 'correlation':
                selected_features = correlation_feature_selection(X_full, y_full, method='pearson', top_k=max_features)
                X_ds = X_full[:, selected_features]

            elif strat == 'random':
                if X_full.shape[1] >= max_features:
                    random_features = np.random.choice(X_full.shape[1], max_features, replace=False)
                else:
                    random_features = np.arange(X_full.shape[1])
                    print(f"Dataset has only {X_full.shape[1]} features. Using all.")
                X_ds = X_full[:, random_features]
            else:
                raise ValueError(f"Unknown strategy: {strat}")
            
            test_on_tabpfn_v2(X_ds, y_full)
            