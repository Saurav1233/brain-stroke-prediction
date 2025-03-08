import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.joblib')

# Extract necessary components
lr_with_smote = model['model']
preprocessor = model['preprocessor']
numerical_cols = model['numerical_cols']
encoded_cols = model['encoded_cols']



# Function to get user input
def get_user_input():
    print("\nEnter patient details for stroke prediction:")

    gender = input("Gender (Male/Female): ").strip().capitalize()
    age = float(input("Age (years): "))
    hypertension = int(input("Hypertension (0 for No, 1 for Yes): "))
    heart_disease = int(input("Heart Disease (0 for No, 1 for Yes): "))
    ever_married = input("Ever Married? (Yes/No): ").strip().capitalize()
    work_type = input("Work Type (Private/Self-employed/Govt_job/Children/Never_worked): ").strip()
    residence_type = input("Residence Type (Urban/Rural): ").strip().capitalize()
    avg_glucose_level = float(input("Average Glucose Level: "))
    bmi = float(input("BMI (Body Mass Index): "))
    smoking_status = input("Smoking Status (never smoked/formerly smoked/smokes/Unknown): ").strip()

    return {
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

# Get input from the user
sample_input = get_user_input()

# Convert input to DataFrame
input_df = pd.DataFrame([sample_input])

# Apply preprocessing
input_df[encoded_cols] = preprocessor.transform(input_df)

# Select required columns
X = input_df[numerical_cols + encoded_cols]

# Make prediction
prediction = lr_with_smote.predict(X)

# Display the result
print(f"\nStroke Prediction: {'Yes, High Risk!' if prediction[0] == 1 else 'No, Low Risk.'}")
