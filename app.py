import streamlit as st
import pandas as pd
import commons  # Make sure to import your module that contains generate_data, perform_feature_selection, perform_diabetes_test, and perform_bgl_test

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.selectbox("Go to", ["Home", "Diabetic Report"])

    if selection == "Home":
        show_home()
    elif selection == "Diabetic Report":
        show_report()

def show_home():
    st.title("GlucoSense - Diabetes and BGL Test")

    # Input fields
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    gender = 0 if gender == "Male" else 1
    heart_rate = st.number_input("Heart Rate", min_value=0)
    max_bp = st.number_input("Max BP", min_value=0)
    min_bp = st.number_input("Min BP", min_value=0)
    spo2 = st.number_input("SPO2", min_value=0, max_value=100)
    body_vitals = {'Age': [age], 'Gender': [gender], 'Heart_Beat': [heart_rate], 'SPO2': [spo2], 'max_BP': [max_bp], 'min_BP': [min_bp]}
    body_vitals = pd.DataFrame(body_vitals)

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if st.button("Submit"):
        if uploaded_file is not None:
            # Read the CSV file
            data = pd.read_csv(uploaded_file, skiprows=3).iloc[:, 1:]

            # Generate data for the test
            test_data = commons.generate_data(data, body_vitals)

            # Perform tests
            reduced_features = commons.perform_feature_selection(test_data)
            diabetes_result = commons.perform_diabetes_test(reduced_features)
            bgl_result = commons.perform_bgl_test(reduced_features)

            # Store results in session state to access in the report page
            st.session_state["name"] = name
            st.session_state["age"] = age
            st.session_state["gender"] = gender
            st.session_state["heart_rate"] = heart_rate
            st.session_state["max_bp"] = max_bp
            st.session_state["min_bp"] = min_bp
            st.session_state["spo2"] = spo2
            st.session_state["diabetes_result"] = diabetes_result
            st.session_state["bgl_result"] = bgl_result

            st.success("Test Completed! Go to the 'Diabetic Report' page to see the results.")
        else:
            st.warning("Please upload a CSV file.")

def show_report():
    st.title("Diabetic Report")

    if "diabetes_result" in st.session_state and "bgl_result" in st.session_state:
        st.write(f"**Name:** {st.session_state['name']}")
        st.write(f"**Age:** {st.session_state['age']}")
        st.write(f"**Gender:** {'Male' if st.session_state['gender'] == 0 else 'Female'}")
        st.write(f"**Heart Rate:** {st.session_state['heart_rate']}")
        st.write(f"**Max BP:** {st.session_state['max_bp']}")
        st.write(f"**Min BP:** {st.session_state['min_bp']}")
        st.write(f"**SPO2:** {st.session_state['spo2']}")
        st.write(f"**Diabetes Test Result:** {st.session_state['diabetes_result']}")
        st.write(f"**BGL Test Result:** {st.session_state['bgl_result']}")

        # Display the results in a visually appealing manner
        st.markdown("### Diabetes Test Result")
        st.write(f"Your Diabetes Test Result indicates that your blood sugar level is: **{st.session_state['diabetes_result']}**")

        st.markdown("### Blood Glucose Level (BGL) Test Result")
        st.write(f"Your Blood Glucose Level (BGL) Test Result is: **{st.session_state['bgl_result']}**")

        # You can add more visualizations or formatting here to make the report more appealing
    else:
        st.warning("Please complete the test on the 'Home' page first.")

if __name__ == "__main__":
    main()
