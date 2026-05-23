"""
Preprocessing Module for EduGuard AI
Handles data cleaning, feature engineering, and preprocessing for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df):
    """
    Clean the dataset by handling missing values and duplicates
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_clean = df.copy()
    
    # Handle missing values (if any)
    df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean


def encode_target(df):
    """
    Encode the target variable (Target) to numeric values
    Dropout -> 0, Enrolled -> 1, Graduate -> 2
    
    Args:
        df (pd.DataFrame): Dataset with Target column
        
    Returns:
        pd.DataFrame: Dataset with encoded target
        LabelEncoder: Fitted label encoder
    """
    df_encoded = df.copy()
    
    # Encode target: Dropout=0, Enrolled=1, Graduate=2
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df_encoded['Target'] = df_encoded['Target'].map(target_mapping)
    
    return df_encoded, target_mapping


def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding
    
    Args:
        df (pd.DataFrame): Dataset with categorical features
        
    Returns:
        pd.DataFrame: Dataset with encoded features
        dict: Dictionary of label encoders for each categorical column
    """
    df_encoded = df.copy()
    
    # Identify categorical columns (excluding Target)
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if 'Target' in categorical_cols:
        categorical_cols.remove('Target')
    
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        pd.DataFrame: Scaled training features
        pd.DataFrame: Scaled test features
        StandardScaler: Fitted scaler
    """
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_for_training(df, test_size=0.2, random_state=42):
    """
    Prepare the dataset for ML training
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, encoders, scaler
    """
    # Clean data
    df_clean = clean_data(df)
    
    # Encode target
    df_encoded, target_mapping = encode_target(df_clean)
    
    # Separate features and target
    X = df_encoded.drop('Target', axis=1)
    y = df_encoded['Target']
    
    # Encode categorical features
    X_encoded, encoders = encode_categorical_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler, target_mapping


def prepare_single_prediction(input_data, encoders, scaler, feature_columns):
    """
    Prepare a single input for prediction
    
    Args:
        input_data (dict): Dictionary of input features
        encoders (dict): Dictionary of label encoders
        scaler (StandardScaler): Fitted scaler
        feature_columns (list): List of feature columns in order
        
    Returns:
        np.array: Prepared input array
    """
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df_input.columns:
            # Handle unseen categories
            try:
                df_input[col] = encoder.transform(df_input[col].astype(str))
            except ValueError:
                # If category not seen during training, use most common
                df_input[col] = encoder.transform([encoder.classes_[0]])
    
    # Ensure all columns are present and in correct order
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Reorder columns
    df_input = df_input[feature_columns]
    
    # Scale features
    df_input_scaled = scaler.transform(df_input)
    
    return df_input_scaled


if __name__ == "__main__":
    # Test preprocessing
    from data.data_loader import load_dataset
    
    df = load_dataset()
    X_train, X_test, y_train, y_test, encoders, scaler, target_mapping = prepare_for_training(df)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target mapping: {target_mapping}")
    print(f"Encoded columns: {list(encoders.keys())}")
