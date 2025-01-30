import streamlit as st
import pandas as pd
import commons  # Make sure to import your module that contains generate_data, perform_feature_selection, perform_diabetes_test, and perform_bgl_test
import seaborn as sns
import matplotlib.pyplot as plt

# Increase the width of the content for all pages
st.set_page_config(layout="wide")

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.selectbox("Go to", ["Personal Information", "Diabetic Report"])

    if selection == "Personal Information":
        show_home()
    elif selection == "Diabetic Report":
        show_report()

def show_home():
    st.title("GlucoSense: A non-invasive diabetes monitor")
    st.write("Enter your personal information below.")
    
    # Creating columns
    col1, col2 = st.columns(2)

    # Input fields with default values in columns
    with col1:
        name = st.text_input("Name", value="Ritik Sharma")
        age = st.number_input("Age", min_value=0, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
        heart_rate = st.number_input("Heart Rate", min_value=0, value=70)
    with col2:
        spo2 = st.number_input("SPO2", min_value=0, max_value=100, value=95)
        min_bp = st.number_input("Min BP", min_value=0, value=80)
        max_bp = st.number_input("Max BP", min_value=0, value=120)
        
        # File upload
        uploaded_file = st.file_uploader("Breath data from GlucoSense", type=["csv"])
    
        if uploaded_file is not None:
            # Read the CSV file
            data = pd.read_csv(uploaded_file, skiprows=3).iloc[:, 1:]
    
            # Generate data for the test
            test_data = commons.generate_data(data, st.session_state["body_vitals"])
    
            # Perform tests
            reduced_features = commons.perform_feature_selection(test_data)
            diabetes_result = commons.perform_diabetes_test(reduced_features)
            bgl_result = commons.perform_bgl_test(reduced_features)
    
            # Store results in session state to access in the report page
            st.session_state["diabetes_result"] = diabetes_result
            st.session_state["bgl_result"] = bgl_result
            st.success("Test Completed! Go to the 'Diabetic Report' page to see the results.")
    
        else:
            st.warning("Please upload a CSV file.")
        
    gender = 0 if gender == "Male" else 1
    body_vitals = {'Age': [age], 'Gender': [gender], 'Heart_Beat': [heart_rate], 'SPO2': [spo2], 'max_BP': [max_bp], 'min_BP': [min_bp]}
    body_vitals = pd.DataFrame(body_vitals)

    # Store personal information in session state
    st.session_state["name"] = name
    st.session_state["age"] = age
    st.session_state["gender"] = gender
    st.session_state["heart_rate"] = heart_rate
    st.session_state["max_bp"] = max_bp
    st.session_state["min_bp"] = min_bp
    st.session_state["spo2"] = spo2
    st.session_state["body_vitals"] = body_vitals

def show_report():
    st.title("GlucoSense: A non-invasive diabetes monitor")
    st.write("Here is your detailed diabetes and BGL report.")

    if "diabetes_result" in st.session_state and "bgl_result" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state['name']}")
            st.write(f"**Age:** {st.session_state['age']}")
            st.write(f"**Gender:** {'Male' if st.session_state['gender'] == 0 else 'Female'}")
            st.metric(label="Max BP", value=f"{st.session_state['max_bp']} mmHg")
            st.metric(label="Min BP", value=f"{st.session_state['min_bp']} mmHg")

        with col2:
            st.metric(label="Heart Rate", value=f"{st.session_state['heart_rate']} bpm")
            st.metric(label="SPO2", value=f"{st.session_state['spo2']} %")
            st.metric(label="BGL Severity", value=st.session_state['diabetes_result'])
            st.metric(label="Blood Glucose Level (mg/dL)", value=f"{st.session_state['bgl_result']} mg/dL")

    else:
        st.warning("Please complete the test on the 'Breath Dataset' page first.")

if __name__ == "__main__":
    main()
