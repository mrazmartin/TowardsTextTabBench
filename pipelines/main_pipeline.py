# 1. user chooses a dataset by name or selects all
# 2. the code will go to the datasets_notebooks directory and run the notebook based on the name -> make use of download_datasets.py
# 3. the code will run embedding functions based on the variable settings in __main__ -> make use of text_processors/preprocess_text.py
# 4. the code will downsample the loaded features by selected strategy -> make use of  feature_selection.py
# 5. the code will train a model out of a small selection of models based on the variable settings in __main__, by default it will do xgboost -> main part of this file, rest imported
import numpy as np
import json
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import os

def setup_project_root(path=None):
    if path:
        project_root = os.path.abspath(path)
        os.makedirs(project_root, exist_ok=True)
    else:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root

def save_partial_results(run_timestamp, partial_results, results_path='results.json'):
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[run_timestamp] = partial_results

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main pipeline for dataset processing and model training.")    
    parser.add_argument('--dataset', type=str, default='it_salary', help='Name of the dataset to process (or a group of datasets).')
    parser.add_argument('--embed_methods', nargs='+', default=['fasttext', 'skrub', 'ag'], help='Methods for text embeddings.') 
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'pkl'], help='Format to save embeddings.')
    parser.add_argument('--project_root', type=Path, default=None, help='Output Path')
    parser.add_argument('--download_datasets', action='store_true', help='Run data preprocessing notebooks')
    parser.add_argument('--generate_embeddings', action='store_true', help='Run data preprocessing notebooks')
    parser.add_argument('--run_pipe', action='store_true', help='Run pipeline')
    parser.add_argument('--eval_method', default= 'tabpfn', choices=['xgb', 'tabpfn'],help='Methods for text embeddings.')
    parser.add_argument('--downsample_methods', nargs='+', default=['t-test', 'anova', 'variance', 'pca', 'correlation', 'shap', 'random'], help='Methods for downsampling.') 
    parser.add_argument('--no_text', action='store_true', help='Drop Text Columns')
    args = parser.parse_args()

    os.environ["PROJECT_ROOT"] = setup_project_root(args.project_root)

    from configs.dataset_configs import get_dataset_list, data_configs
    from pipelines.download_datasets import download_datasets
    from pipelines.embedd_text import embedd_datasets, load_embedded_dataset
    from pipelines.feature_selection import downsample_features
    from pipelines.row_downsampling import downsample_rows_wrapper
    from pipelines.evaluation import tabpfn_v2_eval, xgboost_eval, autogluon_eval 

    RUN_TIMESTAMP = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    run_results = {}
    
    # Block 1: Download and process datasets
    if args.download_datasets:
        download_datasets(args.dataset)

    # Block 2: Take the features labeled as textual and embedd them according to the selected methods
    text_embedding_methods = args.embed_methods # -> we are taking all of them
    save_format = args.save_format  
    
    if args.generate_embeddings:
        embedd_datasets(args.dataset, text_embedding_methods, save_format)

    
    if args.run_pipe:
        dataset_name_list = get_dataset_list(args.dataset)
        for dataset_name in dataset_name_list:
            
            # Block 3: Load the text emebeddings and the original data and merge them together into a single dataframe
            data, bundle = load_embedded_dataset(dataset_name, methods=text_embedding_methods, save_format=save_format)

            loaded_config = bundle['config']
            summary_df = bundle['summary']

            if args.no_text:
                text_columns = summary_df.loc[summary_df['Type'] == 'textual', 'Column Name'].tolist()
                df = bundle['data'].copy()
                df = df.drop(columns=text_columns)
                data = {'no-text': df}

            # Block 4: Downsample and Train
            seed = 0
            ds_rows = 3000
            ds_mode = 'stratified'
            data = downsample_rows_wrapper(data, target_col=loaded_config['target'], task=loaded_config['task'],mode=ds_mode, downsampled_rows=ds_rows, seed=seed)

            start_list = args.downsample_methods
            results_path = "."
            if args.eval_method == "tabpfn":
                max_features = 300
                data = downsample_features(data, config_df=loaded_config, summary_df=summary_df, max_features=max_features, strat_list=start_list)
                tabpfn_results = tabpfn_v2_eval(emb_df=data, config=loaded_config, eval_cvg=data_configs[dataset_name])
                dataset_results = {
                    "tabpfn": tabpfn_results,
                }
                results_path = f"tabpfn_results_{dataset_name}.json"
            elif args.eval_method == "xgb":
                max_features = 300
                data = downsample_features(data, config_df=loaded_config, summary_df=summary_df, max_features=max_features, strat_list=start_list)
                xgb_results = xgboost_eval(emb_df=data, config=loaded_config, eval_cvg=data_configs[dataset_name])
                dataset_results = {
                    "xgb": xgb_results,
                }
                results_path = f"xgb_results_{args.dataset}.json"
            else:
                raise ValueError(f"Unsupported evaluation method: '{args.eval_method}'. Supported methods are: 'tabpfn', 'xgb'.")

            run_results[dataset_name] = dataset_results
            save_partial_results(RUN_TIMESTAMP, run_results,results_path=results_path)


