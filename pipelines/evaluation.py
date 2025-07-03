import os
from tabpfn import TabPFNClassifier, TabPFNRegressor
from xgboost import XGBClassifier, XGBRegressor
from autogluon.tabular import TabularPredictor
import numpy as np
from sklearn.metrics import (
    accuracy_score, log_loss,
    root_mean_squared_error, r2_score
)
from sklearn.model_selection import KFold
from datasets_notebooks.dataloader_functions.utils.log_msgs import info_msg
import pandas as pd

project_root = os.environ["PROJECT_ROOT"]

def tabpfn_v2_eval(emb_df, config, eval_cvg, n_splits=5):
    info_msg(f"Starting TabPFN evaluation with {n_splits} splits", color="blue")
    return generic_evaluator(emb_df, config, eval_cvg, 'tabpfn', n_splits)

def xgboost_eval(emb_df, config, eval_cvg, n_splits=5):
    info_msg(f"Starting XGBoost evaluation with {n_splits} splits", color="blue")
    return generic_evaluator(emb_df, config, eval_cvg, 'xgboost', n_splits)

def autogluon_eval(emb_df, config, eval_cvg, n_splits=5):
    info_msg(f"Starting AutoGluon evaluation with {n_splits} splits", color="blue")
    return generic_evaluator(emb_df, config, eval_cvg, 'autogluon', n_splits)

def compute_metric(y_true, y_pred, y_proba, task_type, eval_cvg=None):
    """
    Compute accuracy and loss metrics based on task type.
    If eval_cvg provides custom loss/acc functions, use them.
    """
    info_msg(f"Computing metrics for {task_type} task", color="cyan")
    
    custom_loss_fn = eval_cvg.get('loss', None) if eval_cvg else None
    custom_acc_fn = eval_cvg.get('accuracy', None) if eval_cvg else None
    
    # Classification
    if task_type == 'clf':
        num_classes = len(np.unique(y_true))
        info_msg(f"Classification task with {num_classes} classes", color="cyan")
        
        if custom_acc_fn:
            info_msg("Using custom accuracy function", color="magenta")
            acc = custom_acc_fn(y_true, y_pred)
        else:
            acc = accuracy_score(y_true, y_pred)

        if custom_loss_fn:
            info_msg("Using custom loss function", color="magenta")
            loss = custom_loss_fn(y_true, y_proba)
        else:
            loss = log_loss(y_true, y_proba)

    # Regression
    else:
        info_msg("Regression task", color="cyan")
        if custom_acc_fn:
            info_msg("Using custom accuracy function", color="magenta")
            acc = custom_acc_fn(y_true, y_pred)
        else:
            acc = r2_score(y_true, y_pred)

        if custom_loss_fn:
            info_msg("Using custom loss function", color="magenta")
            loss = custom_loss_fn(y_true, y_pred)
        else:
            loss = root_mean_squared_error(y_true, y_pred)  # RMSE

    info_msg(f"Computed metrics - Loss: {loss:.4f}, Accuracy: {acc:.4f}", color="green")
    return {
        'loss': loss,
        'accuracy': acc
    }

def evaluate_with_cv(model_fn, X, y, task_type, eval_cvg, n_splits=2):
    """Generic K-fold cross validation evaluation"""
    info_msg(f"Starting {n_splits}-fold cross validation", color="blue")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        info_msg(f"Processing fold {i+1}/{n_splits}", color="cyan")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        info_msg("Training model...", color="cyan")
        model = model_fn(task_type)
        model.fit(X_train.values if hasattr(model, 'fit') else X_train, 
                 y_train.values if hasattr(model, 'fit') else y_train)
        
        info_msg("Making predictions...", color="cyan")
        y_pred = model.predict(X_test.values if hasattr(model, 'fit') else X_test)
        y_proba = model.predict_proba(X_test.values if hasattr(model, 'fit') else X_test) if task_type == 'clf' else None
        
        metric = compute_metric(y_test, y_pred, y_proba, task_type, eval_cvg)
        metrics.append(metric)
        info_msg(f"Fold {i+1} completed - Loss: {metric['loss']:.4f}, Accuracy: {metric['accuracy']:.4f}", color="green")
    
    # Calculate mean and standard deviation for each metric
    avg_metrics = {
        'mean': {k: np.mean([d[k] for d in metrics]) for k in metrics[0]},
        'std': {k: np.std([d[k] for d in metrics]) for k in metrics[0]},
        'var': {k: np.var([d[k] for d in metrics]) for k in metrics[0]},
        'raw_metrics': metrics  # Optionally include all raw metrics
    }
    
    info_msg(
        f"CV completed - Avg Loss: {avg_metrics['mean']['loss']:.4f} (±{avg_metrics['std']['loss']:.4f}), "
        f"Avg Accuracy: {avg_metrics['mean']['accuracy']:.4f} (±{avg_metrics['std']['accuracy']:.4f})", 
        color="green"
    )
    return avg_metrics

def get_tabpfn_model(task_type):
    """Return appropriate TabPFN model based on task type"""
    info_msg(f"Initializing TabPFN model for {task_type} task", color="cyan")
    if task_type == 'clf':
        return TabPFNClassifier(ignore_pretraining_limits=True)
    return TabPFNRegressor(ignore_pretraining_limits=True)

def get_xgboost_model(task_type, num_classes=None):
    """Return appropriate XGBoost model based on task type"""
    info_msg(f"Initializing XGBoost model for {task_type} task", color="cyan")
    if task_type == 'clf':
        if num_classes == 2:
            info_msg("Binary classification detected", color="magenta")
            return XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            info_msg(f"Multi-class classification with {num_classes} classes", color="magenta")
            return XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
    info_msg("Regression task detected", color="magenta")
    return XGBRegressor()

def get_autogluon_predictor(df, target_col, task_type, num_classes=None):
    """Configure and return AutoGluon predictor"""
    problem_type = ('binary' if num_classes == 2 else 'multiclass') if task_type == 'clf' else 'regression'
    info_msg(f"Initializing AutoGluon predictor for {problem_type} problem", color="cyan")
    return TabularPredictor(
        label=target_col,
        problem_type=problem_type,
        eval_metric='log_loss' if task_type == 'clf' else 'rmse',
        path = project_root
    )

def generic_evaluator(emb_df, config, eval_cvg, model_type='tabpfn', n_splits=5):
    """Generic evaluator function with K-fold CV"""
    info_msg(f"Starting generic evaluation with model type: {model_type}", color="blue")
    results = {}
    target_col = config['target']
    task_type = config['task']
    
    for key in emb_df.keys():
        info_msg(f"\nEvaluating dataset: {key}", color="blue")
        df = emb_df[key].copy()
        df = df.astype({col: 'float64' for col in df.columns if pd.api.types.is_int64_dtype(df[col])})
        info_msg(f"Dataset shape: {df.shape}", color="cyan")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]

        num_classes = len(y.unique()) if task_type == 'clf' else None
        
        if model_type == 'tabpfn':
            info_msg("Using TabPFN model", color="magenta")
            metrics = evaluate_with_cv(
                lambda _: get_tabpfn_model(task_type),
                X, y, task_type, eval_cvg, n_splits
            )
        elif model_type == 'xgboost':
            info_msg("Using XGBoost model", color="magenta")
            metrics = evaluate_with_cv(
                lambda _: get_xgboost_model(task_type, num_classes),
                X, y, task_type, eval_cvg, n_splits
            )
        elif model_type == 'autogluon':
            info_msg("Using AutoGluon model", color="magenta")
            fold_metrics = []
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for i, (train_idx, test_idx) in enumerate(kf.split(df)):
                info_msg(f"Processing fold {i+1}/{n_splits}", color="cyan")
                train_data = df.iloc[train_idx]
                test_data = df.iloc[test_idx]

                print(train_data.dtypes)
                
                info_msg("Training AutoGluon predictor...", color="cyan")
                predictor = get_autogluon_predictor(df, target_col, task_type, num_classes)
                predictor.fit(train_data, time_limit=60)
                
                y_test = test_data[target_col]
                y_pred = predictor.predict(test_data.drop(columns=[target_col]))
                y_proba = predictor.predict_proba(test_data.drop(columns=[target_col])) if task_type == 'clf' else None
                
                metric = compute_metric(y_test, y_pred, y_proba, task_type, eval_cvg)
                fold_metrics.append(metric)
                info_msg(f"Fold {i+1} completed - Loss: {metric['loss']:.4f}, Accuracy: {metric['accuracy']:.4f}", color="green")
            
            metrics = {
                'mean': {k: np.mean([d[k] for d in fold_metrics]) for k in fold_metrics[0]},
                'std': {k: np.std([d[k] for d in fold_metrics]) for k in fold_metrics[0]},
                'var': {k: np.var([d[k] for d in fold_metrics]) for k in fold_metrics[0]},
                'raw_metrics': fold_metrics
            }
        
        results[key] = metrics
        info_msg(
            f"Completed evaluation for {key} - "
            f"Loss: {metrics['mean']['loss']:.4f} (±{metrics['std']['loss']:.4f}), "
            f"Accuracy: {metrics['mean']['accuracy']:.4f} (±{metrics['std']['accuracy']:.4f})", 
            color="green"
        )
    
    info_msg(f"\nEvaluation completed for all datasets with {model_type}", color="green")
    return results