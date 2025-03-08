import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Brain Stroke Prediction", page_icon="🧠", layout="wide")

# Load the saved model
@st.cache_resource
def load_model():
    model_data = joblib.load('model.joblib')
    return model_data

# Main function to run the app
def main():
    # Load the model and components
    model_data = load_model()
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    numerical_cols = model_data['numerical_cols']
    categorical_cols = model_data['categorical_cols']
    encoded_cols = model_data['encoded_cols']

    # Title and description
    st.title("🧠 Brain Stroke Prediction App")
    st.markdown("""
    This app predicts the likelihood of a brain stroke based on various health factors.
    Please enter the patient's information below.
    """)

    # Create input form
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0, step=0.1)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])

        with col2:
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=79.0, step=0.1)
            bmi = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

        submit_button = st.form_submit_button(label="Predict")

    # Process prediction when form is submitted
    if submit_button:
        # Create input dictionary
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input
        try:
            # Transform categorical variables
            input_df[encoded_cols] = preprocessor.transform(input_df)

            # Prepare final input for prediction
            X = input_df[numerical_cols + encoded_cols]

            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]

            # Display results
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error(f"⚠️ Stroke Risk Detected - Probability: {probability:.2%}")
            else:
                st.success(f"✅ No Stroke Risk Detected - Probability: {probability:.2%}")

            # Additional information
            with st.expander("See Input Details"):
                st.write("Input Parameters:")
                st.write(input_data)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add some additional information
    st.sidebar.header("About")
    st.sidebar.info("""
    This Brain Stroke Prediction App uses a machine learning model trained on various health factors.
    The model uses Logistic Regression with SMOTE to handle class imbalance.
    
    Features used:
    - Gender
    - Age
    - Hypertension
    - Heart Disease
    - Marital Status
    - Work Type
    - Residence Type
    - Glucose Level
    - BMI
    - Smoking Status
    """)

if __name__ == "__main__":
    main()