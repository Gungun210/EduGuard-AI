"""
Dashboard Component for EduGuard AI
Displays analytics dashboard with KPI cards and interactive charts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_dataset
from model.predict import DropoutPredictor


def create_kpi_card(title, value, subtitle, color, icon):
    """
    Create a styled KPI card HTML
    
    Args:
        title (str): Card title
        value: Main value to display
        subtitle (str): Subtitle/description
        color (str): Color for the card
        icon (str): Icon emoji
        
    Returns:
        str: HTML for the KPI card
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 14px; color: #aaa; margin-bottom: 5px;">{title}</div>
                <div style="font-size: 32px; font-weight: bold; color: #fff;">{value}</div>
                <div style="font-size: 12px; color: #888; margin-top: 5px;">{subtitle}</div>
            </div>
            <div style="font-size: 48px;">{icon}</div>
        </div>
    </div>
    """
    return html


def render_dashboard():
    """
    Render the main dashboard page
    """
    st.title("📊 Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_dataset()
    
    # Calculate KPIs
    total_students = len(df)
    dropout_count = len(df[df['Target'] == 'Dropout'])
    graduate_count = len(df[df['Target'] == 'Graduate'])
    enrolled_count = len(df[df['Target'] == 'Enrolled'])
    
    dropout_rate = (dropout_count / total_students) * 100
    
    # Load model for accuracy
    try:
        predictor = DropoutPredictor()
        model_info = predictor.get_model_info()
        model_accuracy = model_info['accuracy'] * 100
    except:
        model_accuracy = 0
    
    # Estimate at-risk students (Dropout + Enrolled)
    at_risk_students = dropout_count + enrolled_count
    
    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_kpi_card(
            "Total Students",
            f"{total_students:,}",
            "In dataset",
            "#4A90E2",
            "👥"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card(
            "Dropout Rate",
            f"{dropout_rate:.1f}%",
            f"{dropout_count} students",
            "#FF4B4B",
            "⚠️"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_kpi_card(
            "At-Risk Students",
            f"{at_risk_students:,}",
            "Dropout + Enrolled",
            "#FFA500",
            "🎯"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_kpi_card(
            "Model Accuracy",
            f"{model_accuracy:.1f}%",
            "Best performing model",
            "#00C851",
            "🤖"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Student Status Distribution")
        fig_pie = create_pie_chart(dropout_count, graduate_count, enrolled_count)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top 10 Dropout Risk Factors")
        try:
            feature_importance = predictor.feature_importance.head(10)
            fig_bar = create_feature_importance_chart(feature_importance)
            st.plotly_chart(fig_bar, use_container_width=True)
        except:
            st.info("Model not trained yet. Please train the model first.")
    
    st.markdown("---")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Feature Correlation Heatmap")
        fig_heatmap = create_correlation_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.subheader("📈 Dropout Trend by Age Group")
        fig_line = create_age_trend_chart(df)
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Additional Statistics
    st.markdown("---")
    st.subheader("📋 Detailed Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Graduates", f"{graduate_count:,}", f"{(graduate_count/total_students)*100:.1f}%")
    
    with col2:
        st.metric("Currently Enrolled", f"{enrolled_count:,}", f"{(enrolled_count/total_students)*100:.1f}%")
    
    with col3:
        st.metric("Dropouts", f"{dropout_count:,}", f"{(dropout_count/total_students)*100:.1f}%")


def create_pie_chart(dropout, graduate, enrolled):
    """
    Create a pie chart for student status distribution
    
    Args:
        dropout (int): Number of dropouts
        graduate (int): Number of graduates
        enrolled (int): Number of enrolled students
        
    Returns:
        plotly.graph_objects.Figure: Pie chart
    """
    fig = go.Figure(data=[go.Pie(
        labels=['Dropout', 'Graduate', 'Enrolled'],
        values=[dropout, graduate, enrolled],
        marker=dict(colors=['#FF4B4B', '#00C851', '#4A90E2']),
        textinfo='label+percent',
        textfont_size=14,
        hole=0.4,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    return fig


def create_feature_importance_chart(feature_importance):
    """
    Create a bar chart for feature importance
    
    Args:
        feature_importance (pd.DataFrame): Feature importance data
        
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    fig = go.Figure(data=[go.Bar(
        x=feature_importance['importance'].head(10),
        y=feature_importance['feature'].head(10),
        orientation='h',
        marker=dict(
            color=feature_importance['importance'].head(10),
            colorscale='Viridis'
        ),
        text=feature_importance['importance'].head(10).round(3),
        textposition='outside'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        xaxis=dict(
            title='Importance',
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title='Feature',
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        height=400,
        margin=dict(l=200, r=20, t=20, b=20)
    )
    
    return fig


def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for key features
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        plotly.graph_objects.Figure: Heatmap
    """
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to top correlated features to avoid overcrowding
    if len(numeric_cols) > 15:
        # Calculate correlation with target-like columns
        numeric_cols = numeric_cols[:15]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(
            title='Correlation',
            titlefont=dict(color='#ffffff'),
            tickfont=dict(color='#ffffff')
        )
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        xaxis=dict(
            tickfont=dict(size=8, color='#ffffff'),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=8, color='#ffffff'),
            showgrid=False
        ),
        height=400,
        margin=dict(l=100, r=100, t=20, b=100)
    )
    
    return fig


def create_age_trend_chart(df):
    """
    Create a line chart showing dropout trend by age group
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        plotly.graph_objects.Figure: Line chart
    """
    # Create age groups
    df_copy = df.copy()
    df_copy['Age Group'] = pd.cut(
        df_copy['Age at enrollment'],
        bins=[15, 20, 25, 30, 35, 40, 50, 70],
        labels=['15-20', '21-25', '26-30', '31-35', '36-40', '41-50', '50+']
    )
    
    # Calculate dropout rate by age group
    age_dropout = df_copy.groupby('Age Group').apply(
        lambda x: (x['Target'] == 'Dropout').sum() / len(x) * 100
    ).reset_index()
    age_dropout.columns = ['Age Group', 'Dropout Rate']
    
    fig = go.Figure(data=[go.Scatter(
        x=age_dropout['Age Group'],
        y=age_dropout['Dropout Rate'],
        mode='lines+markers',
        line=dict(color='#FF4B4B', width=3),
        marker=dict(size=10, color='#FF4B4B'),
        fill='tozeroy',
        fillcolor='rgba(255, 75, 75, 0.2)'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        xaxis=dict(
            title='Age Group',
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title='Dropout Rate (%)',
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        height=400,
        margin=dict(l=60, r=20, t=20, b=60)
    )
    
    return fig


if __name__ == "__main__":
    render_dashboard()
