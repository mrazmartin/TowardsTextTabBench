import json
import pandas as pd
from pathlib import Path
import numpy as np

def process_json_file(file_path, existing_df=None):
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the run data (assuming there's only one run per file)
    run_data = next(iter(data.values()))
    
    # Prepare a list to store all results
    all_results = []
    
    # Generate all possible column names from embedding types and downsampling techniques
    embedding_types = ['fasttext', 'ag', 'skrub']
    downsampling_techs = ['variance', 'pca', 'correlation', 'shap', 'random', 'all']
    
    # Process each dataset in the file
    for dataset_name in run_data:
        dataset_data = run_data[dataset_name]
        
        # Process each model in the dataset
        for model_name in dataset_data:
            model_data = dataset_data[model_name]
            
            # Prepare a dictionary to store this dataset-model's results
            results = {
                'dataset_name': dataset_name,
                'model': model_name
            }
            
            # Initialize all possible combinations with NaN
            for emb in embedding_types:
                for tech in downsampling_techs:
                    col_name = f"{emb}_df_{tech}"
                    results[col_name] = np.nan
            
            # Fill in the actual values from the JSON
            for metric_key in model_data:
                # Split the key into embedding type and downsampling technique
                parts = metric_key.split('_')
                emb_type = parts[0]
                tech = '_'.join(parts[2:])  # handles cases like 'df_variance' or 'df_all'
                
                # Get the mean accuracy
                # mean_accuracy = model_data[metric_key]['mean']['accuracy']
                std = model_data[metric_key]['std']['accuracy']
                
                # Store in the results dictionary
                col_name = f"{emb_type}_df_{tech}"
                results[col_name] = std
            
            all_results.append(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Merge with existing DataFrame if provided
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True)
    
    return df

def main():
    # Initialize an empty DataFrame
    final_df = None
    
    # Get all JSON files in the current directory (or specify a path)
    json_files =[f for f in Path('.').glob('*.json') if 'xgb' in f.name]
    
    # Process each file
    for json_file in json_files:
        final_df = process_json_file(json_file, final_df)
    
    # Save to CSV
    if final_df is not None:
        final_df.to_csv('xgb_std.csv', index=False)
        print(f"Results saved to xgb_std.csv with {len(final_df)} rows")
    else:
        print("No JSON files found to process.")

if __name__ == '__main__':
    main()