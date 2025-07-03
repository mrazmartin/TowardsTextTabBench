import os
import pandas as pd
import pickle
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
from pipelines.main_pipeline import setup_project_root

os.environ["PROJECT_ROOT"] = setup_project_root("/work/dlclarge2/dasb-Camvid/tabadap")

from pipelines.embedd_text import load_raw_bundle
from pipelines.evaluation import tabpfn_v2_eval
from configs.dataset_configs import data_configs
from pipelines.feature_selection import downsample_features
from pipelines.row_downsampling import downsample_rows_wrapper
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def downsample_data(raw_df: pd.DataFrame, config: Dict[str, Any], target_col: str) -> pd.DataFrame:
    """Downsample the data using stratified sampling."""
    logger.info("Downsampling data...")
    raw_dict = downsample_rows_wrapper(
        {"raw": raw_df},
        target_col=target_col,
        task=config['task'],
        mode='stratified',
        downsampled_rows=3000,
        seed=42
    )
    return raw_dict["raw"]

def process_carte_embeddings(emb_path: Path, target_col: str) -> pd.DataFrame:
    """Load and process CARTE embeddings."""
    logger.info(f"Loading CARTE embeddings from: {emb_path}")
    with open(emb_path, 'rb') as file:
        data = pickle.load(file)
    
    train_df = pd.DataFrame(np.vstack(data["train_features"]))
    train_df[target_col] = data["train_target"]
    
    test_df = pd.DataFrame(np.vstack(data["test_features"]))
    test_df[target_col] = data["test_target"]
    
    return pd.concat([train_df, test_df]).reset_index(drop=True)

def add_bert_embeddings(df: pd.DataFrame, raw_df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """Add BERT embeddings for text columns."""
    logger.info("Adding BERT embeddings...")
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_dfs = []
    
    for col in text_columns:
        texts = raw_df[col].fillna("").astype(str).tolist()
        embeddings = bert.encode(texts, show_progress_bar=True)
        col_embedding_df = pd.DataFrame(
            embeddings, 
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
        )
        embedding_dfs.append(col_embedding_df)
    
    return pd.concat([df.reset_index(drop=True), pd.concat(embedding_dfs, axis=1)], axis=1)


def save_results(results: Dict[str, Any], json_path: str) -> None:
    """Save results to JSON file."""
    logger.info(f"Saving results to {json_path}")
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
            existing_data.update(results)
            results = existing_data
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def create_bert_baseline(raw_df: pd.DataFrame, text_columns: list, non_text_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create BERT baseline by combining BERT embeddings with non-text features."""
    logger.info("Creating BERT + no-text baseline...")
    
    # Get BERT embeddings for text columns
    bert_df = add_bert_embeddings(pd.DataFrame(), raw_df, text_columns)
    
    # Combine with non-text features
    combined_df = pd.concat([bert_df, non_text_df.drop(columns=[target_col], errors='ignore')], axis=1)
    combined_df[target_col] = non_text_df[target_col].values
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Extended Ablations with CARTE embeddings.")
    parser.add_argument('--dataset_name', type=str, default='kickstarter', help='Name of the dataset to process')
    parser.add_argument('--emb_path', type=Path, default="kick_carte_center_node_no_text_embedding.pkl", help='Path to CARTE embeddings')
    parser.add_argument('--no_text_baseline', action='store_true', help='Run no text baseline')
    parser.add_argument('--bert_baseline', action='store_true', help='Run BERT + no-text baseline (no CARTE)')
    parser.add_argument("--add_bert", action='store_true', help='Include BERT embeddings with CARTE')
    parser.add_argument("--include_og_non_text", action='store_true', help='Include original non-textual columns')
    parser.add_argument('--key', type=str, default="carte_exp", help='Experiment Key')

    args = parser.parse_args()

    try:
        # Load data
        bundle = load_raw_bundle(dataset_name=args.dataset_name)
        summary_df = bundle['summary']
        loaded_config = bundle['config']
        target_col = loaded_config["target"]
        raw_df = bundle['data'].copy()
        text_columns = summary_df.loc[summary_df['Type'] == 'textual', 'Column Name'].tolist()

        # Downsample data
        raw_df = downsample_data(raw_df, loaded_config, target_col)
        no_text_df = raw_df.drop(columns=text_columns)

        if args.no_text_baseline:
            logger.info("Running no-text baseline...")
            tabpfn_results = tabpfn_v2_eval(
                emb_df={'no_text_baseline': no_text_df},
                config=loaded_config,
                eval_cvg=data_configs[args.dataset_name]
            )
        elif args.bert_baseline:
            logger.info("Running BERT + no-text baseline...")
            bert_baseline_df = create_bert_baseline(raw_df, text_columns, no_text_df, target_col)
            emb_df = downsample_features(
                {"bert_baseline_df": bert_baseline_df},
                config_df=loaded_config,
                summary_df=summary_df,
                max_features=301,
                strat_list=["shap"]
            )
            tabpfn_results = tabpfn_v2_eval(
                emb_df=emb_df,
                config=loaded_config,
                eval_cvg=data_configs[args.dataset_name]
            )
        else:
            # Process CARTE embeddings
            carte_df = process_carte_embeddings(args.emb_path, target_col)

            if args.include_og_non_text:
                logger.info("Including original non-textual columns...")
                no_text_df = no_text_df.drop(columns=[target_col], errors='ignore')
                carte_df = pd.concat([carte_df, no_text_df], axis=1)
                target_summary = summary_df
            else:
                target_summary = summary_df[summary_df['Column Name'] == target_col]

            if args.add_bert:
                carte_df = add_bert_embeddings(carte_df, raw_df, text_columns)

            # Create experiment key
            key = args.key
            # Downsample features and evaluate
            emb_df = downsample_features(
                {key: carte_df},
                config_df=loaded_config,
                summary_df=target_summary,
                max_features=301,
                strat_list=["shap"]
            )
            tabpfn_results = tabpfn_v2_eval(
                emb_df=emb_df,
                config=loaded_config,
                eval_cvg=data_configs[args.dataset_name]
            )

        # Save results
        save_results(tabpfn_results, "extended_ablations_carte.json")
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()