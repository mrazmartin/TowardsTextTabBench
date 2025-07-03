import os, sys
import pandas as pd
from datasets_notebooks.text_processors.preprocess_text import generate_text_embeddings, load_text_embeddings, embeddings_to_df
from configs.dataset_configs import get_dataset_list, get_a_dataset_dict
from datasets_notebooks.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg
from sklearn.preprocessing import OrdinalEncoder

project_root = os.environ["PROJECT_ROOT"]

def build_embedding_path(dataset_name: str) -> str:
    """
    Build the path to save the embeddings for a given dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
    
    Returns:
        str: Path to save the embeddings.
    """
    datasets_dir = os.path.join(project_root, 'datasets_files', 'embeddings')
    dataset_config = get_a_dataset_dict(dataset_name)
    
    if dataset_config['task'] == 'clf':
        task_folder = 'classification'
    elif dataset_config['task'] == 'reg':
        task_folder = 'regression'
    else:
        raise ValueError(f"Unknown task type '{dataset_config['task']}' for dataset '{dataset_name}'.")

    if dataset_config['name']:
        dataset_folder_name = dataset_config['name']

    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    
    embedding_path = os.path.join(datasets_dir, task_folder, dataset_folder_name)
    
    return embedding_path

def build_pckl_path(dataset_name: str) -> str:
    """
    Build the path to load the raw data for a given dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
    
    Returns:
        str: Path to load the raw data.
    """
    datasets_dir = os.path.join(project_root, 'datasets_files', 'raw')
    dataset_config = get_a_dataset_dict(dataset_name)
    
    if dataset_config['task'] == 'clf':
        task_folder = 'classification'
    elif dataset_config['task'] == 'reg':
        task_folder = 'regression'
    else:
        raise ValueError(f"Unknown task type '{dataset_config['task']}' for dataset '{dataset_name}'.")

    if dataset_config['name']:
        dataset_folder_name = dataset_config['name']

    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    
    pckl_path = os.path.join(datasets_dir, task_folder, dataset_folder_name, f"{dataset_name}_processed.pkl")
    
    return pckl_path

def embedd_datasets(
        datasets_selection, 
        methods=['fasttext', 'ag', 'skrub'], 
        save_format='npy',
        output_path=None
        ):
    """
    Embedd the datasets using the selected methods.
    """

    dataset_name_list = get_dataset_list(datasets_selection)

    for dataset_name in dataset_name_list:
        # 1. load the raw data
        load_data_pth = build_pckl_path(dataset_name)
        if output_path:
            load_data_pth = os.path.join(output_path,load_data_pth)
        
        bundle = pd.read_pickle(load_data_pth)
        loaded_df = bundle['data']
        loaded_df.columns = loaded_df.columns.get_level_values(0)
        summary_df = bundle['summary']
        loaded_config = bundle['config']

        # 2. embedd and save the data
        save_embedding_path = build_embedding_path(dataset_name)
        if output_path:
            save_embedding_path = os.path.join(output_path,save_embedding_path)
        
        embeddings = generate_text_embeddings(
            df=loaded_df, 
            meta_df=summary_df, 
            emb_path=save_embedding_path, 
            methods=methods, 
            save_format=save_format
        )

def process_cats(df,summary_df):
    cat_cols = summary_df.loc[summary_df['Type'] == 'categorical', 'Column Name'].tolist()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df

def load_embeddings(dataset_name, methods):
    emb_path = build_embedding_path(dataset_name)
    embeddings = load_text_embeddings(emb_path=emb_path, methods=methods)
    return embeddings

def load_raw_bundle(dataset_name):
    load_data_pth = build_pckl_path(dataset_name)
    bundle = pd.read_pickle(load_data_pth)
    bundle['data'].columns = bundle['data'].columns.get_level_values(0)
    bundle['data'] = process_cats(bundle['data'], bundle['summary'])
    return bundle

def load_embedded_dataset(dataset_name, methods=['fasttext', 'ag', 'skrub'], save_format='npy'):
    """
    Get the embedded dataset for a given dataset name.
    """
    # 1. load the embeddings
    embeddings = load_embeddings(dataset_name, methods)
    # 2. load the raw data
    bundle = load_raw_bundle(dataset_name)
    # 3. convert to df
    dfs = embeddings_to_df(embeddings, original_df=bundle["data"], meta_df=bundle["summary"])
    return dfs, bundle

if __name__ == "__main__":

    # BLock = 2 =
    # Take the features labeled as textual and embedd them according to the selected methods

    # 1. load the pp data:
    dataset_name = 'hs_cards'

    if False:
        embeddings = embedd_datasets(
            datasets_selection=dataset_name, 
            methods=['ag'], 
            save_format='npy'
        )
    else:

        df_path = build_pckl_path(dataset_name)
        bundle = pd.read_pickle(df_path)
        # 1. Extract components
        loaded_df = bundle['data']
        loaded_df.columns = loaded_df.columns.get_level_values(0)
        summary_df = bundle['summary']
        loaded_config = bundle['config']

        # 2. embedd and save the data
        generate_text_embeddings(
            df=loaded_df, 
            meta_df=summary_df, 
            emb_path=build_embedding_path(dataset_name),
            methods=['skrub'],
            save_format='npy'
        )
        # 3. load embeddings
        embeddings = load_text_embeddings(emb_path=build_embedding_path(dataset_name))

        # 4. convert to df
        dfs = embeddings_to_df(embeddings, original_df=loaded_df, meta_df=summary_df)

        # 5. aesthetic printing ✨
        for df_name, df in dfs.items():
            print(f"======{df_name}======")
            print(df.head(3))