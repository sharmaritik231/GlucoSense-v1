import streamlit as st
import pandas as pd
import commons
import seaborn as sns
import matplotlib.pyplot as plt

# Custom CSS styling
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background-image: url('https://img.freepik.com/free-vector/blue-healthcare-pattern_53876-93017.jpg');
            background-size: cover;
            background-attachment: fixed;
        }
        
        .stApp {
            background-color: rgba(255, 255, 255, 0.9);
        }
        
        .stSidebar {
            background-color: #f0f9ff !important;
        }
        
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #2a9df4;
        }
        
        .title-text {
            color: #1a4b8e;
            font-weight: 600;
            margin-bottom: 30px;
        }
        
        .upload-section {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .result-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)

# Increase the width of the content for all pages
st.set_page_config(layout="wide", page_icon="🩺")
inject_custom_css()

def main():
    st.sidebar.title("🩺 GlucoSense Navigation")
    selection = st.sidebar.radio("Go to", ["Body Vitals", "Report"], label_visibility="collapsed")

    if selection == "Body Vitals":
        show_home()
    elif selection == "Report":
        show_report()

def show_home():
    st.markdown('<h1 class="title-text">GlucoSense: AI-Powered Diabetes Monitoring</h1>', unsafe_allow_html=True)
    
    # Hero Section
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div style="background: #e3f2fd; padding: 25px; border-radius: 15px; margin-bottom: 30px;">
                <h3 style="color: #1a4b8e; margin-top: 0;">🌐 Non-Invasive Diabetes Prediction</h3>
                <p style="font-size: 0.95em;">GlucoSense revolutionizes diabetes monitoring through advanced breath analysis and AI-driven insights. Our system combines:</p>
                <ul>
                    <li>🔬 VOC Pattern Recognition</li>
                    <li>❤️ Real-time Physiological Monitoring</li>
                    <li>📊 Predictive Analytics Engine</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Input Section
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("📝 Patient Information")
            name = st.text_input("Full Name", value="Ritik Sharma")
            age = st.number_input("Age", min_value=0, value=30)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
            
        with col2:
            st.subheader("🩺 Vital Signs")
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, value=70)
            spo2 = st.number_input("SPO2 (%)", min_value=0, max_value=100, value=95)
            min_bp, max_bp = st.columns(2)
            with min_bp:
                min_bp = st.number_input("Diastolic BP", min_value=0, value=80)
            with max_bp:
                max_bp = st.number_input("Systolic BP", min_value=0, value=120)
    
    # Data Upload Section
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Breath Data Upload")
        uploaded_file = st.file_uploader("Upload GlucoSense Sensor Data (CSV format)", type=["csv"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, skiprows=3).iloc[:, 1:]
            test_data = commons.generate_data(data, st.session_state["body_vitals"])
            reduced_features = commons.perform_feature_selection(test_data)
            diabetes_result = commons.perform_diabetes_test(reduced_features)
            bgl_result = commons.perform_bgl_test(reduced_features)
            
            st.session_state["diabetes_result"] = diabetes_result
            st.session_state["bgl_result"] = bgl_result
            
            st.success("✅ Analysis Complete! Navigate to the Report section for results.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Store session data
    gender_code = 0 if gender == "Male" else 1
    body_vitals = pd.DataFrame({
        'Age': [age], 'Gender': [gender_code], 
        'Heart_Beat': [heart_rate], 'SPO2': [spo2],
        'max_BP': [max_bp], 'min_BP': [min_bp]
    })
    st.session_state.update({
        "name": name, "age": age, "gender": gender,
        "heart_rate": heart_rate, "max_bp": max_bp,
        "min_bp": min_bp, "spo2": spo2,
        "body_vitals": body_vitals
    })

def show_report():
    st.markdown('<h1 class="title-text">Comprehensive Diabetes Analysis Report</h1>', unsafe_allow_html=True)
    
    if "diabetes_result" not in st.session_state:
        st.warning("⚠️ Please complete the analysis on the Body Vitals page first.")
        return
    
    # Patient Summary
    with st.container():
        st.subheader("👤 Patient Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #2a9df4; font-size: 0.9em;">Patient Name</div>
                <div style="font-size: 1.4em; font-weight: 600;">{st.session_state['name']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #2a9df4; font-size: 0.9em;">Age | Gender</div>
                <div style="font-size: 1.4em; font-weight: 600;">
                    {st.session_state['age']} | {st.session_state['gender']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #2a9df4; font-size: 0.9em;">Blood Pressure</div>
                <div style="font-size: 1.4em; font-weight: 600;">
                    {st.session_state['max_bp']}/{st.session_state['min_bp']} mmHg
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Health Metrics
    with st.container():
        st.subheader("📈 Health Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        status_color = {
            "Non-Diabetic": "#4CAF50",
            "Prediabetic": "#FFC107",
            "Highly Diabetic": "#F44336"
        }.get(st.session_state['diabetes_result'], "#2a9df4")
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {status_color};">
                <div style="color: #2a9df4; font-size: 0.9em;">Diabetes Status</div>
                <div style="font-size: 1.4em; font-weight: 600; color: {status_color};">
                    {st.session_state['diabetes_result']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #2a9df4; font-size: 0.9em;">Blood Glucose</div>
                <div style="font-size: 1.4em; font-weight: 600;">
                    {st.session_state['bgl_result']} mg/dL
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #2a9df4; font-size: 0.9em;">Heart Rate</div>
                <div style="font-size: 1.4em; font-weight: 600;">
                    {st.session_state['heart_rate']} bpm
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #2a9df4; font-size: 0.9em;">Blood Oxygen</div>
                <div style="font-size: 1.4em; font-weight: 600;">
                    {st.session_state['spo2']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations Section
    with st.container():
        st.subheader("📋 Medical Recommendations")
        if "Highly Diabetic" in st.session_state['diabetes_result']:
            rec_color = "#F44336"
            recommendations = """
            - Immediate consultation with endocrinologist required
            - Schedule fasting blood glucose test
            - Begin lifestyle modification program
            """
        elif "Prediabetic" in st.session_state['diabetes_result']:
            rec_color = "#FFC107"
            recommendations = """
            - Regular glucose monitoring advised
            - Dietary consultation recommended
            - Increase physical activity
            """
        else:
            rec_color = "#4CAF50"
            recommendations = """
            - Maintain healthy lifestyle
            - Annual diabetes screening
            - Balanced diet recommendation
            """
        
        st.markdown(f"""
        <div style="background: {rec_color}10; padding: 20px; border-radius: 10px; border-left: 4px solid {rec_color}; margin-top: 20px;">
            <h4 style="color: {rec_color}; margin-top: 0;">Clinical Guidance</h4>
            <div style="line-height: 1.8;">
                {recommendations}
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
