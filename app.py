import streamlit as st
import pandas as pd
import commons  # Make sure to import your module that contains generate_data, perform_diabetes_test, and perform_bgl_test

def main():
    st.title("GlucoSense - Diabetes and BGL Test")

    # Input fields
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    heart_rate = st.number_input("Heart Rate", min_value=0)
    max_bp = st.number_input("Max BP", min_value=0)
    min_bp = st.number_input("Min BP", min_value=0)
    spo2 = st.number_input("SPO2", min_value=0.0, max_value=100.0, format="%.1f")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if st.button("Submit"):
        if uploaded_file is not None:
            # Read the CSV file
            data = pd.read_csv(uploaded_file, skiprows=3).iloc[:, 1:]

            # Generate data for the test
            data['Age'] = age
            data['Gender'] = gender
            data['Heart_Beat'] = heart_rate
            data['SPO2'] = spo2
            data['max_BP'] = max_bp
            data['min_BP'] = min_bp
            test_data = commons.generate_data(data)

            # Perform tests
            diabetes_result = commons.perform_diabetes_test(test_data)
            bgl_result = commons.perform_bgl_test(test_data)

            # Display results
            st.write(f"Name: {name}")
            st.write(f"Age: {age}")
            st.write(f"Gender: {gender}")
            st.write(f"Heart Rate: {heart_rate}")
            st.write(f"Max BP: {max_bp}")
            st.write(f"Min BP: {min_bp}")
            st.write(f"SPO2: {spo2}")
            st.write(f"Diabetes Test Result: {diabetes_result}")
            st.write(f"BGL Test Result: {bgl_result}")
        else:
            st.warning("Please upload a CSV file.")

if __name__ == "__main__":
    main()
