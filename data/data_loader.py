"""
Data Loader Module for EduGuard AI
Loads the Predict Students Dropout and Academic Success dataset from UCI ML Repository
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo
import os


def load_dataset():
    """
    Load the Predict Students Dropout and Academic Success dataset from UCI ML Repository
    Dataset ID: 697
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    try:
        # Fetch dataset from UCI ML Repository
        dataset = fetch_ucirepo(id=697)
        
        # Get features and target as pandas dataframes
        X = dataset.data.features
        y = dataset.data.targets
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def get_dataset_info(df):
    """
    Get basic information about the dataset
    
    Args:
        df (pd.DataFrame): The dataset
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['Target'].value_counts().to_dict() if 'Target' in df.columns else None
    }
    return info


if __name__ == "__main__":
    # Test the data loader
    df = load_dataset()
    info = get_dataset_info(df)
    
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Shape: {info['shape']}")
    print(f"\nColumns: {info['columns']}")
    print(f"\nMissing Values: {info['missing_values']}")
    print(f"\nTarget Distribution: {info['target_distribution']}")
