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
    
    # Extract dataset name and model name
    dataset_name = next(iter(run_data.keys()))
    model_name = next(iter(run_data[dataset_name].keys()))
    
    # Get the metrics data
    metrics_data = run_data[dataset_name][model_name]
    
    # Prepare a dictionary to store all the results
    results = {
        'dataset_name': dataset_name,
        'model': model_name
    }
    
    # Generate all possible column names from embedding types and downsampling techniques
    embedding_types = ['fasttext', 'ag', 'skrub']
    downsampling_techs = ['t-test', 'anova', 'variance', 'pca', 'correlation', 'shap', 'random']
    
    # Initialize all possible combinations with NaN
    for emb in embedding_types:
        for tech in downsampling_techs:
            col_name = f"{emb}_df_{tech}"
            results[col_name] = np.nan
    
    # Fill in the actual values from the JSON
    for metric_key in metrics_data:
        # Split the key into embedding type and downsampling technique
        parts = metric_key.split('_')
        emb_type = parts[0]
        tech = '_'.join(parts[2:])  # handles cases like 'df_variance'
        
        # Get the mean accuracy
        std = metrics_data[metric_key]['std']['accuracy']
        
        # Store in the results dictionary
        col_name = f"{emb_type}_df_{tech}"
        results[col_name] = std
    
    # Convert to DataFrame
    df = pd.DataFrame([results])
    
    # Merge with existing DataFrame if provided
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True)
    
    return df

def main():
    # Initialize an empty DataFrame
    final_df = None
    
    # Get all JSON files in the current directory (or specify a path)
    json_files = [f for f in Path('.').glob('*.json') if 'tabpfn' in f.name]
    
    # Process each file
    for json_file in json_files:
        final_df = process_json_file(json_file, final_df)
    
    # Save to CSV
    if final_df is not None:
        final_df.to_csv('tabpfn_results.csv', index=False)
        print("Results saved to tabpfn_results.csv")
    else:
        print("No JSON files found to process.")

if __name__ == '__main__':
    main()
