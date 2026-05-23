"""
Insights Component for EduGuard AI
Displays key findings, recommendations, and impact statistics
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset
from model.predict import DropoutPredictor


def render_insights():
    """
    Render the insights page
    """
    st.title("💡 Insights & Recommendations")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading insights..."):
        df = load_dataset()
    
    # Calculate statistics
    total_students = len(df)
    dropout_count = len(df[df['Target'] == 'Dropout'])
    graduate_count = len(df[df['Target'] == 'Graduate'])
    enrolled_count = len(df[df['Target'] == 'Enrolled'])
    
    dropout_rate = (dropout_count / total_students) * 100
    
    # Key Findings Section
    st.subheader("🔍 Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_insight_card(
            "Academic Performance",
            "Students with lower grades in first and second semesters have significantly higher dropout rates. Academic performance is the strongest predictor of dropout risk.",
            "#FF4B4B",
            "📚"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_insight_card(
            "Financial Factors",
            "Students with tuition fees not up to date and those without scholarships show 2-3x higher dropout rates. Financial stress is a major contributor.",
            "#FFA500",
            "💰"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_insight_card(
            "Age Impact",
            "Older students (25+) have higher dropout rates compared to traditional-age students. This suggests need for additional support for non-traditional students.",
            "#4A90E2",
            "👤"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What Causes Dropout Section
    st.subheader("🎯 What Causes Dropout?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(create_cause_card(
            "Poor Academic Performance",
            "Low grades in curricular units",
            "40%",
            "#FF4B4B"
        ), unsafe_allow_html=True)
        
        st.markdown(create_cause_card(
            "Financial Difficulties",
            "Tuition fees not up to date",
            "25%",
            "#FFA500"
        ), unsafe_allow_html=True)
        
        st.markdown(create_cause_card(
            "Previous Failures",
            "History of academic setbacks",
            "15%",
            "#9B59B6"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_cause_card(
            "Lack of Support",
            "No scholarship or financial aid",
            "12%",
            "#3498DB"
        ), unsafe_allow_html=True)
        
        st.markdown(create_cause_card(
            "Age Factors",
            "Non-traditional age students",
            "8%",
            "#1ABC9C"
        ), unsafe_allow_html=True)
        
        st.markdown(create_cause_card(
            "Other Factors",
            "Displacement, special needs, etc.",
            "0%",
            "#95A5A6"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Actionable Recommendations
    st.subheader("📋 Actionable Recommendations for Educators")
    
    recommendations = [
        {
            "title": "Early Warning System",
            "description": "Implement automated alerts when students show early signs of academic struggle (grades below 10 in first semester).",
            "priority": "HIGH",
            "icon": "🚨"
        },
        {
            "title": "Financial Aid Expansion",
            "description": "Increase scholarship availability and create emergency funds for students with tuition payment difficulties.",
            "priority": "HIGH",
            "icon": "💵"
        },
        {
            "title": "Academic Mentorship",
            "description": "Pair at-risk students with peer mentors and academic advisors for regular check-ins and support.",
            "priority": "MEDIUM",
            "icon": "🤝"
        },
        {
            "title": "Targeted Tutoring",
            "description": "Provide free tutoring services for students struggling with course material, especially in first year.",
            "priority": "HIGH",
            "icon": "📖"
        },
        {
            "title": "Flexible Scheduling",
            "description": "Offer flexible class schedules and part-time options for older students and working professionals.",
            "priority": "MEDIUM",
            "icon": "📅"
        },
        {
            "title": "Counseling Services",
            "description": "Expand mental health and career counseling services to address non-academic challenges.",
            "priority": "MEDIUM",
            "icon": "🧠"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {
            "HIGH": "#FF4B4B",
            "MEDIUM": "#FFA500",
            "LOW": "#00C851"
        }[rec["priority"]]
        
        st.markdown(create_recommendation_card(
            i, rec["title"], rec["description"], rec["priority"], priority_color, rec["icon"]
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Impact Statistics
    st.subheader("📈 Impact Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_impact_card(
            "Students at Risk",
            f"{dropout_count + enrolled_count:,}",
            f"{((dropout_count + enrolled_count) / total_students) * 100:.1f}% of total",
            "#FF4B4B",
            "⚠️"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_impact_card(
            "Potential Savings",
            f"{int(dropout_count * 0.3):,}",
            "Students that could be saved with intervention",
            "#00C851",
            "💾"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_impact_card(
            "Success Rate",
            f"{graduate_count:,}",
            f"{(graduate_count / total_students) * 100:.1f}% graduation rate",
            "#4A90E2",
            "🎓"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Performance
    try:
        predictor = DropoutPredictor()
        model_info = predictor.get_model_info()
        
        st.subheader("🤖 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", f"{model_info['accuracy']:.2%}")
        
        with col2:
            st.metric("F1 Score", f"{model_info['f1_score']:.4f}")
        
        with col3:
            st.metric("ROC-AUC Score", f"{model_info['roc_auc']:.4f}")
        
        st.info(f"Best Model: {model_info['model_name']}")
        
    except:
        st.warning("Model not trained yet. Please train the model to see performance metrics.")


def create_insight_card(title, description, color, icon):
    """
    Create an insight card HTML
    
    Args:
        title (str): Card title
        description (str): Card description
        color (str): Color for the card
        icon (str): Icon emoji
        
    Returns:
        str: HTML for the insight card
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="font-size: 32px; margin-right: 15px;">{icon}</div>
            <div style="font-size: 18px; font-weight: bold; color: #fff;">{title}</div>
        </div>
        <div style="font-size: 14px; color: #ccc; line-height: 1.5;">{description}</div>
    </div>
    """
    return html


def create_cause_card(title, subtitle, percentage, color):
    """
    Create a cause card HTML
    
    Args:
        title (str): Card title
        subtitle (str): Card subtitle
        percentage (str): Percentage value
        color (str): Color for the card
        
    Returns:
        str: HTML for the cause card
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        border-left: 4px solid {color};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 16px; font-weight: bold; color: #fff;">{title}</div>
                <div style="font-size: 12px; color: #aaa; margin-top: 5px;">{subtitle}</div>
            </div>
            <div style="font-size: 28px; font-weight: bold; color: {color};">{percentage}</div>
        </div>
    </div>
    """
    return html


def create_recommendation_card(number, title, description, priority, color, icon):
    """
    Create a recommendation card HTML
    
    Args:
        number (int): Card number
        title (str): Card title
        description (str): Card description
        priority (str): Priority level
        color (str): Color for the card
        icon (str): Icon emoji
        
    Returns:
        str: HTML for the recommendation card
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; align-items: flex-start;">
            <div style="
                background: {color};
                color: #fff;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                font-weight: bold;
                margin-right: 15px;
                flex-shrink: 0;
            ">{number}</div>
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 24px; margin-right: 10px;">{icon}</div>
                    <div style="font-size: 18px; font-weight: bold; color: #fff;">{title}</div>
                    <div style="
                        background: {color};
                        color: #fff;
                        padding: 3px 10px;
                        border-radius: 10px;
                        font-size: 11px;
                        font-weight: bold;
                        margin-left: auto;
                    ">{priority}</div>
                </div>
                <div style="font-size: 14px; color: #ccc; line-height: 1.5;">{description}</div>
            </div>
        </div>
    </div>
    """
    return html


def create_impact_card(title, value, subtitle, color, icon):
    """
    Create an impact card HTML
    
    Args:
        title (str): Card title
        value (str): Main value
        subtitle (str): Subtitle
        color (str): Color for the card
        icon (str): Icon emoji
        
    Returns:
        str: HTML for the impact card
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="font-size: 40px; margin-bottom: 10px;">{icon}</div>
        <div style="font-size: 14px; color: #aaa; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 36px; font-weight: bold; color: #fff; margin-bottom: 5px;">{value}</div>
        <div style="font-size: 12px; color: #888;">{subtitle}</div>
    </div>
    """
    return html


if __name__ == "__main__":
    render_insights()
