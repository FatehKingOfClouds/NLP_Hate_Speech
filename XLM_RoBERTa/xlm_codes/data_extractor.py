import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def data_collector(file_names, params, is_train):
    """
    Load and filter datasets by language and training mode
    
    Params:
    - file_names: list of file paths
    - params: dictionary containing 'language', 'how_train', 'sample_ratio', 'take_ratio'
    - is_train: whether loading training data
    
    Returns:
    - df: filtered and sampled DataFrame
    """
    
    all_dfs = []
    
    for file in file_names:
        df = pd.read_csv(file)
        
        # Extract language from filename
        filename = os.path.basename(file)
        # filename format: Language_id_full.csv or Language_id_translated.csv
        language_in_file = filename.split('_')[0]
        
        # Filter by language based on training mode
        if params['how_train'] == 'baseline':
            # Load only the target language
            if language_in_file == params['language']:
                all_dfs.append(df)
        
        elif params['how_train'] == 'all':
            # Load all languages regardless
            all_dfs.append(df)
        
        elif params['how_train'] == 'zero_shot':
            # Load all except the target language
            if language_in_file != params['language']:
                all_dfs.append(df)
        
        elif params['how_train'] == 'all_but_one':
            # Load all languages except one (all_but_one is set in params)
            if 'all_but_one_lang' in params:
                if language_in_file != params['all_but_one_lang']:
                    all_dfs.append(df)
            else:
                all_dfs.append(df)
    
    # Concatenate all dataframes
    if not all_dfs:
        print(f"Warning: No data files found for language {params['language']}")
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Sample data if specified
    if is_train and params.get('sample_ratio', 1.0) < 1.0:
        if params.get('take_ratio', True):
            # sample_ratio is a ratio (0.0 to 1.0)
            df = df.sample(frac=params['sample_ratio'], random_state=42)
        else:
            # sample_ratio is absolute number of samples
            sample_size = min(int(params['sample_ratio']), len(df))
            df = df.sample(n=sample_size, random_state=42)
    
    return df

def load_data(files_path, params, is_train=True):
    """
    Load data files using glob pattern
    
    Params:
    - files_path: path to search for files
    - params: contains 'csv_file' pattern and language info
    - is_train: whether loading training data
    
    Returns:
    - DataFrame with loaded data
    """
    
    # Get all matching files
    if isinstance(files_path, str):
        file_names = glob.glob(files_path)
    else:
        file_names = files_path
    
    if not file_names:
        print(f"Warning: No files found matching pattern {files_path}")
        return pd.DataFrame()
    
    # Use data_collector to filter and process
    df = data_collector(file_names, params, is_train)
    
    return df
