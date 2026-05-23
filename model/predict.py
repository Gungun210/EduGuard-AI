"""
Prediction Module for EduGuard AI
Handles model inference and risk prediction
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import prepare_single_prediction


class DropoutPredictor:
    """
    Predictor class for student dropout risk assessment
    """
    
    def __init__(self, model_path=None, artifacts_path=None):
        """
        Initialize the predictor with trained model and artifacts
        
        Args:
            model_path (str): Path to saved model
            artifacts_path (str): Path to saved artifacts
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'saved_models', 'best_model.pkl')
        if artifacts_path is None:
            artifacts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        'saved_models', 'artifacts.pkl')
        
        # Load model and artifacts
        self.model = joblib.load(model_path)
        self.artifacts = joblib.load(artifacts_path)
        
        self.encoders = self.artifacts['encoders']
        self.scaler = self.artifacts['scaler']
        self.feature_names = self.artifacts['feature_names']
        self.target_mapping = self.artifacts['target_mapping']
        self.feature_importance = self.artifacts['feature_importance']
        self.best_model_name = self.artifacts['best_model_name']
        self.metrics = self.artifacts['metrics']
        
        # Reverse target mapping for interpretation
        self.reverse_target_mapping = {v: k for k, v in self.target_mapping.items()}
    
    def predict(self, input_data):
        """
        Make prediction for a single student
        
        Args:
            input_data (dict): Dictionary of student features
            
        Returns:
            dict: Prediction results with risk level and probability
        """
        # Prepare input
        X = prepare_single_prediction(input_data, self.encoders, self.scaler, self.feature_names)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get predicted class
        predicted_class = self.reverse_target_mapping[prediction]
        
        # Get dropout probability (class 0)
        dropout_prob = probabilities[0] * 100
        
        # Determine risk level
        if dropout_prob >= 60:
            risk_level = "HIGH"
            risk_color = "#FF4B4B"
        elif dropout_prob >= 30:
            risk_level = "MEDIUM"
            risk_color = "#FFA500"
        else:
            risk_level = "LOW"
            risk_color = "#00C851"
        
        # Get top risk factors for this student
        risk_factors = self._get_personal_risk_factors(input_data)
        
        # Get recommendations
        recommendations = self._get_recommendations(predicted_class, dropout_prob, risk_factors)
        
        return {
            'prediction': predicted_class,
            'dropout_probability': dropout_prob,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probabilities': {
                'Dropout': probabilities[0] * 100,
                'Enrolled': probabilities[1] * 100,
                'Graduate': probabilities[2] * 100
            },
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    
    def _get_personal_risk_factors(self, input_data, top_n=3):
        """
        Get top risk factors for a specific student based on feature importance
        
        Args:
            input_data (dict): Student input data
            top_n (int): Number of top factors to return
            
        Returns:
            list: List of top risk factors with descriptions
        """
        # Get top features from feature importance
        top_features = self.feature_importance.head(top_n)
        
        risk_factors = []
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Get value for this student
            value = input_data.get(feature, 'N/A')
            
            # Create description based on feature
            description = self._get_feature_description(feature, value)
            
            risk_factors.append({
                'feature': feature,
                'value': value,
                'importance': importance,
                'description': description
            })
        
        return risk_factors
    
    def _get_feature_description(self, feature, value):
        """
        Get human-readable description for a feature value
        
        Args:
            feature (str): Feature name
            value: Feature value
            
        Returns:
            str: Description
        """
        descriptions = {
            'Age at enrollment': f"Age: {value}",
            'Admission grade': f"Admission grade: {value}",
            'Curricular units 1st sem (grade)': f"1st semester grade: {value}",
            'Curricular units 2nd sem (grade)': f"2nd semester grade: {value}",
            'Tuition fees up to date': f"Tuition status: {'Up to date' if value == 1 else 'Not up to date'}",
            'Scholarship holder': f"Scholarship: {'Yes' if value == 1 else 'No'}",
            'Debtor': f"Debtor status: {'Yes' if value == 1 else 'No'}",
            'Gender': f"Gender: {'Male' if value == 1 else 'Female'}",
            'Displaced': f"Displaced: {'Yes' if value == 1 else 'No'}",
            'Educational special needs': f"Special needs: {'Yes' if value == 1 else 'No'}"
        }
        
        return descriptions.get(feature, f"{feature}: {value}")
    
    def _get_recommendations(self, prediction, dropout_prob, risk_factors):
        """
        Get personalized recommendations based on prediction
        
        Args:
            prediction (str): Predicted class
            dropout_prob (float): Dropout probability
            risk_factors (list): Top risk factors
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        if dropout_prob >= 60:
            recommendations.append("⚠️ Immediate intervention required")
            recommendations.append("Schedule one-on-one counseling session")
            recommendations.append("Review financial aid options")
            recommendations.append("Assign academic mentor")
        elif dropout_prob >= 30:
            recommendations.append("📊 Monitor academic progress closely")
            recommendations.append("Schedule regular check-ins")
            recommendations.append("Provide additional study resources")
        else:
            recommendations.append("✅ Student on track for success")
            recommendations.append("Continue current support programs")
            recommendations.append("Encourage participation in enrichment activities")
        
        # Add specific recommendations based on risk factors
        for factor in risk_factors:
            if 'grade' in factor['feature'].lower():
                if factor['value'] < 10:
                    recommendations.append("📚 Provide tutoring support for low grades")
            if 'tuition' in factor['feature'].lower() and factor['value'] == 0:
                recommendations.append("💰 Address tuition payment issues")
            if 'scholarship' in factor['feature'].lower() and factor['value'] == 0:
                recommendations.append("🎓 Explore scholarship opportunities")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': self.best_model_name,
            'accuracy': self.metrics['accuracy'],
            'f1_score': self.metrics['f1_score'],
            'roc_auc': self.metrics['roc_auc'],
            'feature_importance': self.feature_importance
        }


if __name__ == "__main__":
    # Test the predictor
    predictor = DropoutPredictor()
    
    # Sample input
    sample_input = {
        'Marital status': 1,
        'Application mode': 1,
        'Application order': 1,
        'Course': 1,
        'Daytime/evening attendance': 1,
        'Previous qualification': 1,
        'Previous qualification (grade)': 120,
        'Nacionality': 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 1,
        "Father's occupation": 1,
        'Admission grade': 120,
        'Displaced': 1,
        'Educational special needs': 0,
        'Debtor': 0,
        'Tuition fees up to date': 1,
        'Gender': 1,
        'Scholarship holder': 0,
        'Age at enrollment': 20,
        'International': 0,
        'Curricular units 1st sem (credited)': 0,
        'Curricular units 1st sem (enrolled)': 6,
        'Curricular units 1st sem (evaluations)': 6,
        'Curricular units 1st sem (approved)': 5,
        'Curricular units 1st sem (grade)': 12,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)': 0,
        'Curricular units 2nd sem (enrolled)': 6,
        'Curricular units 2nd sem (evaluations)': 6,
        'Curricular units 2nd sem (approved)': 5,
        'Curricular units 2nd sem (grade)': 13,
        'Curricular units 2nd sem (without evaluations)': 0,
        'Unemployment rate': 10,
        'Inflation rate': 2,
        'GDP': 1
    }
    
    result = predictor.predict(sample_input)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Prediction: {result['prediction']}")
    print(f"Dropout Probability: {result['dropout_probability']:.2f}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nTop Risk Factors:")
    for factor in result['risk_factors']:
        print(f"  - {factor['description']}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
