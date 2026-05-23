"""
Predictor Component for EduGuard AI
Live prediction interface with input form and risk visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import DropoutPredictor


def render_predictor():
    """
    Render the live prediction page
    """
    st.title("🔮 Live Dropout Prediction")
    st.markdown("---")
    
    # Load predictor
    try:
        predictor = DropoutPredictor()
    except Exception as e:
        st.error(f"Model not found. Please train the model first. Error: {e}")
        st.info("Run 'python model/train_model.py' to train the model.")
        return
    
    # Create input form
    st.subheader("📝 Enter Student Information")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age at Enrollment", min_value=15, max_value=70, value=20)
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widower", "Divorced", "Facto Union"])
            scholarship = st.selectbox("Scholarship Holder", ["No", "Yes"])
            tuition_up_to_date = st.selectbox("Tuition Fees Up to Date", ["Yes", "No"])
            debtor = st.selectbox("Debtor", ["No", "Yes"])
            displaced = st.selectbox("Displaced", ["No", "Yes"])
            special_needs = st.selectbox("Educational Special Needs", ["No", "Yes"])
            international = st.selectbox("International", ["No", "Yes"])
        
        with col2:
            admission_grade = st.slider("Admission Grade (0-200)", 0, 200, 120)
            prev_qual_grade = st.slider("Previous Qualification Grade (0-200)", 0, 200, 120)
            course = st.selectbox("Course", [
                "Biofuel Production Technologies",
                "Animation and Multimedia Design",
                "Social Service (evening attendance)",
                "Agronomy",
                "Communication Design",
                "Veterinary Nursing",
                "Informatics Engineering",
                "Equinculture",
                "Management",
                "Social Service",
                "Tourism",
                "Nursing",
                "Oral Hygiene",
                "Advertising and Marketing Management",
                "Journalism and Communication",
                "Basic Education",
                "Management (evening attendance)"
            ])
            daytime_attendance = st.selectbox("Daytime/Evening Attendance", ["Daytime", "Evening"])
            prev_qualification = st.selectbox("Previous Qualification", [
                "Secondary Education",
                "Higher Education - Bachelor's",
                "Higher Education - Degree",
                "Higher Education - Master's",
                "Higher Education - Doctorate",
                "Frequency of Higher Education",
                "Other"
            ])
            application_mode = st.selectbox("Application Mode", [
                "1st phase - general contingent",
                "2nd phase - general contingent",
                "3rd phase - general contingent",
                "Over 23 years old",
                "Change of course",
                "Technological specialization",
                "Other"
            ])
        
        with col3:
            # Semester 1 grades
            st.markdown("### Semester 1")
            sem1_enrolled = st.number_input("Curricular Units Enrolled (Sem 1)", min_value=0, max_value=20, value=6)
            sem1_approved = st.number_input("Curricular Units Approved (Sem 1)", min_value=0, max_value=20, value=5)
            sem1_grade = st.slider("Average Grade (Sem 1)", 0, 20, 12)
            sem1_evaluations = st.number_input("Evaluations (Sem 1)", min_value=0, max_value=20, value=6)
            
            # Semester 2 grades
            st.markdown("### Semester 2")
            sem2_enrolled = st.number_input("Curricular Units Enrolled (Sem 2)", min_value=0, max_value=20, value=6)
            sem2_approved = st.number_input("Curricular Units Approved (Sem 2)", min_value=0, max_value=20, value=5)
            sem2_grade = st.slider("Average Grade (Sem 2)", 0, 20, 13)
            sem2_evaluations = st.number_input("Evaluations (Sem 2)", min_value=0, max_value=20, value=6)
        
        # Economic indicators
        st.markdown("### Economic Indicators")
        col1, col2, col3 = st.columns(3)
        with col1:
            unemployment_rate = st.slider("Unemployment Rate (%)", 0, 20, 10)
        with col2:
            inflation_rate = st.slider("Inflation Rate (%)", 0, 10, 2)
        with col3:
            gdp = st.slider("GDP Growth (%)", -5, 10, 1)
        
        submitted = st.form_submit_button("🔍 Predict Dropout Risk", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_data = prepare_input_dict(
            age, gender, marital_status, scholarship, tuition_up_to_date,
            debtor, displaced, special_needs, international, admission_grade,
            prev_qual_grade, course, daytime_attendance, prev_qualification,
            application_mode, sem1_enrolled, sem1_approved, sem1_grade,
            sem1_evaluations, sem2_enrolled, sem2_approved, sem2_grade,
            sem2_evaluations, unemployment_rate, inflation_rate, gdp
        )
        
        # Make prediction
        with st.spinner("Analyzing student data..."):
            result = predictor.predict(input_data)
        
        # Display results
        display_prediction_results(result)


def prepare_input_dict(age, gender, marital_status, scholarship, tuition_up_to_date,
                       debtor, displaced, special_needs, international, admission_grade,
                       prev_qual_grade, course, daytime_attendance, prev_qualification,
                       application_mode, sem1_enrolled, sem1_approved, sem1_grade,
                       sem1_evaluations, sem2_enrolled, sem2_approved, sem2_grade,
                       sem2_evaluations, unemployment_rate, inflation_rate, gdp):
    """
    Prepare input dictionary for prediction
    
    Returns:
        dict: Input data dictionary
    """
    # Map categorical values to numeric
    gender_map = {"Female": 0, "Male": 1}
    yes_no_map = {"No": 0, "Yes": 1}
    marital_map = {"Single": 1, "Married": 2, "Widower": 3, "Divorced": 4, "Facto Union": 5}
    course_map = {
        "Biofuel Production Technologies": 33,
        "Animation and Multimedia Design": 171,
        "Social Service (evening attendance)": 8014,
        "Agronomy": 9003,
        "Communication Design": 9070,
        "Veterinary Nursing": 9085,
        "Informatics Engineering": 9119,
        "Equinculture": 9130,
        "Management": 9147,
        "Social Service": 9238,
        "Tourism": 9254,
        "Nursing": 9500,
        "Oral Hygiene": 9556,
        "Advertising and Marketing Management": 9670,
        "Journalism and Communication": 9773,
        "Basic Education": 9853,
        "Management (evening attendance)": 9991
    }
    prev_qual_map = {
        "Secondary Education": 1,
        "Higher Education - Bachelor's": 2,
        "Higher Education - Degree": 3,
        "Higher Education - Master's": 4,
        "Higher Education - Doctorate": 5,
        "Frequency of Higher Education": 6,
        "Other": 9
    }
    app_mode_map = {
        "1st phase - general contingent": 1,
        "2nd phase - general contingent": 2,
        "3rd phase - general contingent": 5,
        "Over 23 years old": 10,
        "Change of course": 15,
        "Technological specialization": 16,
        "Other": 17
    }
    
    input_data = {
        'Marital status': marital_map[marital_status],
        'Application mode': app_mode_map.get(application_mode, 17),
        'Application order': 1,
        'Course': course_map.get(course, 9147),
        'Daytime/evening attendance': 1 if daytime_attendance == "Daytime" else 0,
        'Previous qualification': prev_qual_map.get(prev_qualification, 9),
        'Previous qualification (grade)': prev_qual_grade,
        'Nacionality': 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 1,
        "Father's occupation": 1,
        'Admission grade': admission_grade,
        'Displaced': yes_no_map[displaced],
        'Educational special needs': yes_no_map[special_needs],
        'Debtor': yes_no_map[debtor],
        'Tuition fees up to date': yes_no_map[tuition_up_to_date],
        'Gender': gender_map[gender],
        'Scholarship holder': yes_no_map[scholarship],
        'Age at enrollment': age,
        'International': yes_no_map[international],
        'Curricular units 1st sem (credited)': 0,
        'Curricular units 1st sem (enrolled)': sem1_enrolled,
        'Curricular units 1st sem (evaluations)': sem1_evaluations,
        'Curricular units 1st sem (approved)': sem1_approved,
        'Curricular units 1st sem (grade)': sem1_grade,
        'Curricular units 1st sem (without evaluations)': 0,
        'Curricular units 2nd sem (credited)': 0,
        'Curricular units 2nd sem (enrolled)': sem2_enrolled,
        'Curricular units 2nd sem (evaluations)': sem2_evaluations,
        'Curricular units 2nd sem (approved)': sem2_approved,
        'Curricular units 2nd sem (grade)': sem2_grade,
        'Curricular units 2nd sem (without evaluations)': 0,
        'Unemployment rate': unemployment_rate,
        'Inflation rate': inflation_rate,
        'GDP': gdp
    }
    
    return input_data


def display_prediction_results(result):
    """
    Display prediction results with visualizations
    
    Args:
        result (dict): Prediction results
    """
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Risk Level Card
    risk_color = result['risk_color']
    risk_level = result['risk_level']
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}44 100%);
            border: 3px solid {risk_color};
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        ">
            <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">RISK LEVEL</div>
            <div style="font-size: 48px; font-weight: bold; color: {risk_color};">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Gauge chart for dropout probability
        fig_gauge = create_gauge_chart(result['dropout_probability'], risk_color)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4A90E222 0%, #4A90E244 100%);
            border: 2px solid #4A90E2;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        ">
            <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">PREDICTION</div>
            <div style="font-size: 36px; font-weight: bold; color: #4A90E2;">{result['prediction']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Probability Breakdown")
        prob_data = pd.DataFrame({
            'Category': list(result['probabilities'].keys()),
            'Probability': list(result['probabilities'].values())
        })
        
        fig_prob = px.bar(
            prob_data,
            x='Category',
            y='Probability',
            color='Category',
            color_discrete_map={
                'Dropout': '#FF4B4B',
                'Enrolled': '#4A90E2',
                'Graduate': '#00C851'
            },
            text='Probability'
        )
        
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prob.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            yaxis=dict(
                title='Probability (%)',
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff')
            ),
            xaxis=dict(
                tickfont=dict(color='#ffffff')
            ),
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Top Risk Factors")
        for i, factor in enumerate(result['risk_factors'], 1):
            st.markdown(f"""
            <div style="
                background: rgba(255, 75, 75, 0.1);
                border-left: 4px solid #FF4B4B;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            ">
                <div style="font-size: 16px; font-weight: bold; color: #fff;">
                    {i}. {factor['description']}
                </div>
                <div style="font-size: 12px; color: #aaa; margin-top: 5px;">
                    Importance: {factor['importance']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Personalized Recommendations")
    
    for i, rec in enumerate(result['recommendations'], 1):
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00C85122 0%, #00C85144 100%);
            border: 2px solid #00C851;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 16px; color: #fff;">{rec}</div>
        </div>
        """, unsafe_allow_html=True)


def create_gauge_chart(probability, color):
    """
    Create a gauge chart for dropout probability
    
    Args:
        probability (float): Dropout probability
        color (str): Color for the gauge
        
    Returns:
        plotly.graph_objects.Figure: Gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Dropout Probability (%)", 'font': {'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ffffff'},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': '#ffffff',
            'steps': [
                {'range': [0, 30], 'color': '#00C85133'},
                {'range': [30, 60], 'color': '#FFA50033'},
                {'range': [60, 100], 'color': '#FF4B4B33'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        },
        number={'font': {'color': '#ffffff', 'size': 40}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


if __name__ == "__main__":
    render_predictor()
