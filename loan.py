import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model
filename = 'voting_clf_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create Streamlit app
st.title("Loan Default Prediction App")

# Input fields for user data
st.header("Enter Loan Applicant Information")

loan_amount = st.number_input("Loan Amount")
term = st.number_input("Loan Term (in months)")
credit_history = st.number_input("Credit History")
income = st.number_input("Annual Income")
age = st.number_input("Age")
employment_length = st.number_input("Employment Length (in years)")
loan_purpose = st.selectbox("Loan Purpose", ['Debt Consolidation', 'Home Improvement', 'Business Loan', 'Personal Loan', 'Other'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])

# Create a button to predict
if st.button("Predict Loan Default"):
  # Create a DataFrame from the user input
  new_data = pd.DataFrame({
      'Loan_Amount': [loan_amount],
      'Term': [term],
      'Credit_History': [credit_history],
      'Annual_Income': [income],
      'Age': [age],
      'Employment_Length': [employment_length],
      'Loan_Purpose': [loan_purpose],
      'Marital_Status': [marital_status]
  })

  # Preprocess the data
  le = LabelEncoder()
  new_data['Loan_Purpose'] = le.fit_transform(new_data['Loan_Purpose'])
  new_data['Marital_Status'] = le.fit_transform(new_data['Marital_Status'])


  # Make a prediction using the loaded model
  prediction = loaded_model.predict(new_data)

  # Display the prediction
  if prediction[0] == 0:
    st.success("Loan is predicted to be not in default.")
  else:
    st.error("Loan is predicted to be in default.")