"""
Model Training Module for EduGuard AI
Trains multiple ML models and selects the best performing one
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset
from utils.preprocessing import prepare_for_training


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        RandomForestClassifier: Trained model
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    """
    Train XGBoost Classifier
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        XGBClassifier: Trained model
    """
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)
    return xgb


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression Classifier
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        LogisticRegression: Trained model
    """
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        multi_class='multinomial'
    )
    lr.fit(X_train, y_train)
    return lr


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC-AUC (if available)
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        roc_auc = 0.0
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return metrics


def get_feature_importance(model, feature_names, model_name):
    """
    Get feature importance from the model
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model
        
    Returns:
        pd.DataFrame: DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance


def train_and_select_best_model():
    """
    Train multiple models and select the best one based on accuracy
    
    Returns:
        dict: Dictionary containing best model, metrics, and artifacts
    """
    print("="*60)
    print("TRAINING ML MODELS FOR EDUGUARD AI")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading dataset...")
    df = load_dataset()
    
    print("\nPreparing data for training...")
    X_train, X_test, y_train, y_test, encoders, scaler, target_mapping = prepare_for_training(df)
    
    feature_names = X_train.columns.tolist()
    
    # Train models
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Compare models
    models_metrics = [rf_metrics, xgb_metrics, lr_metrics]
    models_metrics.sort(key=lambda x: x['accuracy'], reverse=True)
    
    best_model_info = models_metrics[0]
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_info['model_name']}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"F1 Score: {best_model_info['f1_score']:.4f}")
    print(f"ROC-AUC: {best_model_info['roc_auc']:.4f}")
    print("="*60)
    
    # Select best model
    if best_model_info['model_name'] == "Random Forest":
        best_model = rf_model
    elif best_model_info['model_name'] == "XGBoost":
        best_model = xgb_model
    else:
        best_model = lr_model
    
    # Get feature importance
    feature_importance = get_feature_importance(best_model, feature_names, best_model_info['model_name'])
    
    # Create model directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and artifacts
    model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    
    artifacts_path = os.path.join(model_dir, 'artifacts.pkl')
    artifacts = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_mapping': target_mapping,
        'feature_importance': feature_importance,
        'best_model_name': best_model_info['model_name'],
        'metrics': best_model_info
    }
    joblib.dump(artifacts, artifacts_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Artifacts saved to: {artifacts_path}")
    
    return {
        'model': best_model,
        'metrics': best_model_info,
        'artifacts': artifacts,
        'feature_importance': feature_importance
    }


if __name__ == "__main__":
    results = train_and_select_best_model()
    print("\nTraining complete!")
