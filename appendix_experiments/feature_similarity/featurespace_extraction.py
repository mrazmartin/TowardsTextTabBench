import os
import json
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_loaders.load_raw import load_raw_data_as_df, delete_data
from utils.handle_files import read_yaml
from configs.directory import config_directory
from carte_pp import preprocess_data as preprocess_data_carte


def load_dataset_names():
    config_path = config_directory['dataset_config']
    data_config_yaml = read_yaml(config_path)
    carte_datasets_list = data_config_yaml['carte_datasets']
    return carte_datasets_list

## Data Preprocessing Functions
def _drop_high_null(data, proportion=0.5):
    """Drop columns with high fraction of missing values"""
    null_num = np.array([data[col].isnull().sum() for col in data.columns])
    null_crit = int(len(data) * proportion)
    null_col = list(data.columns[null_num > null_crit])
    return data.drop(columns=null_col)

def _drop_single_unique(data):
    """ Drops columns that have only a single unique value. Handles dictionary columns safely. """

    # Convert dictionary-like columns to strings
    for col in data.columns:
        if data[col].apply(lambda x: isinstance(x, dict)).any():
            print(f"🛠 Converting column '{col}' from dict to JSON string")
            data[col] = data[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

    # Now safely calculate nunique()
    num_unique_cols = [col for col in data.columns if data[col].nunique() == 1]
    print(f"🗑 Dropping columns with a single unique value: {num_unique_cols}")
    
    return data.drop(columns=num_unique_cols)

def _drop_id_columns(data):
    """Drop columns with id in the name."""
    # drop cols with stand-alone id
    id_cols = [col for col in data.columns if 'id' in col.lower()]
    # drop cols with '_id'
    id_cols += [col for col in data.columns if '_id' in col.lower()]
    # drop cols with 'id_'
    id_cols += [col for col in data.columns if 'id_' in col.lower()]
    # drop 'url' columns
    id_cols += [col for col in data.columns if 'url' in col.lower()]

    return data.drop(columns=id_cols)

## Dataset Loading Functions
def load_dataset_no_pp(dataset_name):
    """
    Loading the data, we want to free-up the memory.
    """
    df_dict, data_config = load_raw_data_as_df(dataset_name)
    file_path = data_config['files'][0]
    file_name = file_path.split('/')[-1]

    # do some very STANDARD pp:
    # drop everything that start with 'Unnamed'
    df_dict[file_name] = df_dict[file_name].loc[:, ~df_dict[file_name].columns.str.startswith('Unnamed')]
    # drop every column with less than half of the data:
    df_dict[file_name] = _drop_high_null(df_dict[file_name])
    # drop every column with single unique values:
    df_dict[file_name] = _drop_single_unique(df_dict[file_name])
    # drop every column with 'id' in the name:
    df_dict[file_name] = _drop_id_columns(df_dict[file_name])

    feature_names = df_dict[file_name].columns.tolist()

    return feature_names, data_config

def load_and_delete_dataset(dataset_name):
    """
    Loading the data, we want to free-up the memory.
    """

    feature_names, data_config = load_dataset_no_pp(dataset_name)

    # now delete the data:
    delete_data(dataset_name, data_config)

    return feature_names

def run_over_all_datasets(output_path='datasets_features.json'):
    """
    Assumes that there is either only one file,\
    or that the first file is the one to use (all files have same features).
    """
    datasets = load_dataset_names()
    feature_dict = {}
    for dataset in datasets:
        feature_names = load_and_delete_dataset(dataset)
        feature_dict[dataset] = feature_names
        print(f"Dataset: {dataset}\tFeatures: {feature_names}")
    
    with open(output_path, 'w') as f:
        json.dump(feature_dict, f, indent=4)

def run_over_all_datasets_carte_pp(output_path):
    """
    Assumes that there is either only one file,
    or that the first file is the one to use (all files have same features).
    Now also saves an additional JSON that includes an example value
    for each feature of each dataset. For each feature, if a value is missing 
    (None or NaN), the function continues down the column until a non-missing value is found.
    """
    import pandas as pd  # Ensure pandas is imported for null checks

    datasets = load_dataset_names()
    feature_dict = {}
    example_feature_dict = {}
    
    for dataset_name in datasets:
        df_dict, data_cofnig = load_raw_data_as_df(dataset_name)
        file_path = data_cofnig['files'][0]
        file_name = file_path.split('/')[-1]
    
        data_df = df_dict[file_name]
    
        # Remove 'carte_' from the dataset name for preprocessing
        dataset_name_ = dataset_name.replace('carte_', '')
    
        # Preprocess the data using carte_pp
        data_name, data, target_name, entity_name, task, repeated = preprocess_data_carte(data_df, dataset_name_)
    
        feature_names = data.columns.tolist()
        feature_dict[dataset_name] = feature_names

        # Build a dictionary with an example value for each feature.
        # If a value is missing (None or NaN), iterate until a non-missing value is found.
        example_features = {}
        for feature in feature_names:
            example_value = None
            for val in data[feature]:
                if pd.notnull(val):  # Skip over None or NaN values
                    example_value = val
                    break
            # Convert numpy data types to native Python types if necessary.
            if example_value is not None:
                if isinstance(example_value, (np.integer, np.floating)):
                    example_value = example_value.item()
                elif isinstance(example_value, np.ndarray):
                    example_value = example_value.tolist()
            example_features[feature] = example_value

        example_feature_dict[dataset_name] = example_features

        # Clean up the loaded data to free memory.
        delete_data(dataset_name, data_cofnig)
        print(f"Dataset: {dataset_name}\tFeatures: {feature_names}")
    
    # Save the features JSON as before.
    with open(output_path, 'w') as f:
        json.dump(feature_dict, f, indent=4)
    
    # Derive a new output path for the examples JSON (e.g., datasets_features_carte_examples.json)
    base, ext = os.path.splitext(output_path)
    example_output_path = base + '_examples' + ext
    with open(example_output_path, 'w') as f:
        json.dump(example_feature_dict, f, indent=4)
    
    print(f"Saved features JSON to: {output_path}")
    print(f"Saved examples JSON to: {example_output_path}")

if __name__ == '__main__':

    curren_path = os.getcwd()

    if False:
        save_path = os.path.join(curren_path, 'datasets_features_default.json')
        run_over_all_datasets(save_path)
    else:
        save_path = os.path.join(curren_path, 'json_files/datasets_features_carte.json')
        run_over_all_datasets_carte_pp(save_path)