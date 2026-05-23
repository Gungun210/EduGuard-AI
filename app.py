"""
EduGuard AI - Student Dropout Prediction System
Main Streamlit Application
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.dashboard import render_dashboard
from components.predictor import render_predictor
from components.insights import render_insights


# Custom CSS for professional dark theme
def load_custom_css():
    """
    Load custom CSS styling for the application
    """
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #9B59B6;
        --accent-color: #00C851;
        --warning-color: #FFA500;
        --danger-color: #FF4B4B;
        --bg-dark: #0E1117;
        --bg-card: #1E2130;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
        border-right: 2px solid rgba(74, 144, 226, 0.3);
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        background: transparent;
    }
    
    /* Navigation buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #9B59B6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.6);
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    h2, h3 {
        color: #ffffff;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 32px;
        font-weight: bold;
    }
    
    [data-testid="stMetricDelta"] {
        color: #00C851;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div > select {
        background-color: #1E2130;
        color: #ffffff;
        border: 1px solid rgba(74, 144, 226, 0.5);
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1E2130;
        color: #ffffff;
        border: 1px solid rgba(74, 144, 226, 0.5);
        border-radius: 8px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #1E2130;
    }
    
    /* Info boxes */
    .stAlert {
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.2) 0%, rgba(155, 89, 182, 0.2) 100%);
        border: 1px solid rgba(74, 144, 226, 0.5);
        border-radius: 10px;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 200, 81, 0.2) 0%, rgba(0, 200, 81, 0.1) 100%);
        border: 1px solid #00C851;
        border-radius: 10px;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 165, 0, 0.2) 0%, rgba(255, 165, 0, 0.1) 100%);
        border: 1px solid #FFA500;
        border-radius: 10px;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 75, 75, 0.1) 100%);
        border: 1px solid #FF4B4B;
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #1E2130;
        border: 2px dashed rgba(74, 144, 226, 0.5);
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4A90E2 0%, #9B59B6 100%);
    }
    
    /* Spinner */
    .stSpinner {
        color: #4A90E2;
    }
    
    /* Footer */
    footer {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f23 100%);
        padding: 20px;
        text-align: center;
        color: #888;
        border-top: 2px solid rgba(74, 144, 226, 0.3);
    }
    
    /* Hide default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4A90E2 0%, #9B59B6 100%);
        border-radius: 5px;
    }
    
    /* Animation for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """
    Render the application header with logo and branding
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #4A90E2 0%, #9B59B6 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 30px rgba(74, 144, 226, 0.4);
        text-align: center;
    ">
        <h1 style="
            color: white;
            margin: 0;
            font-size: 48px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        ">🎓 EduGuard AI</h1>
        <p style="
            color: rgba(255,255,255,0.9);
            margin: 10px 0 0 0;
            font-size: 18px;
        ">Student Dropout Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Live Demo Banner
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #00C851 0%, #00A86B 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 200, 81, 0.4);
        animation: pulse 2s infinite;
    ">
        <span style="
            color: white;
            font-size: 20px;
            font-weight: bold;
        ">🔴 LIVE DEMO - Competition Entry</span>
    </div>
    
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_footer():
    """
    Render the application footer
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f23 100%);
        padding: 20px;
        margin-top: 40px;
        border-radius: 15px;
        text-align: center;
        border-top: 2px solid rgba(74, 144, 226, 0.3);
    ">
        <p style="color: #888; margin: 0;">
            🏆 EduGuard AI - Data Science Competition Entry 2026
        </p>
        <p style="color: #666; margin: 5px 0 0 0; font-size: 12px;">
            Built with ❤️ using Streamlit, Scikit-learn, XGBoost & Plotly
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_about_page():
    """
    Render the About page
    """
    st.title("ℹ️ About EduGuard AI")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.2) 0%, rgba(155, 89, 182, 0.2) 100%);
            border: 2px solid rgba(74, 144, 226, 0.5);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">🎯 Project Overview</h3>
            <p style="color: #ccc; line-height: 1.6;">
                EduGuard AI is a cutting-edge machine learning system designed to predict student dropout risk 
                in higher education institutions. By analyzing academic performance, financial factors, and 
                demographic data, our system helps educators identify at-risk students early and implement 
                targeted interventions to improve retention rates.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0, 200, 81, 0.2) 0%, rgba(0, 200, 81, 0.1) 100%);
            border: 2px solid rgba(0, 200, 81, 0.5);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">📊 Dataset</h3>
            <p style="color: #ccc; line-height: 1.6;">
                We use the <strong>Predict Students Dropout and Academic Success</strong> dataset from the 
                UCI Machine Learning Repository (ID: 697). The dataset contains information on students enrolled 
                in undergraduate degrees, including demographic data, academic performance, and economic indicators.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255, 165, 0, 0.2) 0%, rgba(255, 165, 0, 0.1) 100%);
            border: 2px solid rgba(255, 165, 0, 0.5);
            border-radius: 15px;
            padding: 25px;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">🤖 Machine Learning Models</h3>
            <p style="color: #ccc; line-height: 1.6;">
                Our system trains and compares three state-of-the-art machine learning algorithms:
            </p>
            <ul style="color: #ccc; line-height: 1.8;">
                <li><strong>Random Forest</strong> - Ensemble learning with decision trees</li>
                <li><strong>XGBoost</strong> - Gradient boosting framework</li>
                <li><strong>Logistic Regression</strong> - Linear classification model</li>
            </ul>
            <p style="color: #ccc; line-height: 1.6; margin-top: 10px;">
                The best-performing model is automatically selected based on accuracy, F1-score, and ROC-AUC metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(155, 89, 182, 0.2) 0%, rgba(155, 89, 182, 0.1) 100%);
            border: 2px solid rgba(155, 89, 182, 0.5);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">👥 Team</h3>
            <p style="color: #ccc; line-height: 1.6;">
                EduGuard AI Team<br>
                Data Science Competition 2026
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.2) 0%, rgba(74, 144, 226, 0.1) 100%);
            border: 2px solid rgba(74, 144, 226, 0.5);
            border-radius: 15px;
            padding: 25px;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">🛠️ Technology Stack</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
                <span style="
                    background: rgba(74, 144, 226, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">Python</span>
                <span style="
                    background: rgba(155, 89, 182, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">Streamlit</span>
                <span style="
                    background: rgba(0, 200, 81, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">Scikit-learn</span>
                <span style="
                    background: rgba(255, 165, 0, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">XGBoost</span>
                <span style="
                    background: rgba(255, 75, 75, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">Plotly</span>
                <span style="
                    background: rgba(52, 152, 219, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">Pandas</span>
                <span style="
                    background: rgba(46, 204, 113, 0.3);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: #fff;
                    font-size: 12px;
                ">NumPy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """
    Main application function
    """
    # Page configuration
    st.set_page_config(
        page_title="EduGuard AI - Student Dropout Prediction",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Text visibility fixes
    st.markdown("""
    <style>
    /* All labels */
    label, .stSelectbox label, .stSlider label, 
    .stNumberInput label, .stRadio label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Tab text */
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    /* Active tab */
    .stTabs [aria-selected="true"] {
        color: #60A5FA !important;
        border-bottom: 3px solid #60A5FA !important;
    }

    /* Slider values */
    .stSlider p {
        color: #D1D5DB !important;
    }

    /* Section headers */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #D1D5DB !important;
    }

    /* General text */
    p, span, div {
        color: #E5E7EB !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load custom CSS
    load_custom_css()
    
    # Render header
    render_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4A90E2 0%, #9B59B6 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
        ">
            <h2 style="color: white; margin: 0;">🎓 EduGuard AI</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "📍 Navigate",
            ["📊 Dashboard", "🔮 Live Prediction", "💡 Insights", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="
            background: rgba(74, 144, 226, 0.1);
            border: 1px solid rgba(74, 144, 226, 0.3);
            border-radius: 10px;
            padding: 15px;
        ">
            <p style="color: #888; font-size: 12px; margin: 0;">
                🏆 Competition Entry 2026
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Render selected page
    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "🔮 Live Prediction":
        render_predictor()
    elif page == "💡 Insights":
        render_insights()
    elif page == "ℹ️ About":
        render_about_page()
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
