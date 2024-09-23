# Importing all necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Loading the trained model
model = tf.keras.models.load_model('model.h5')

# Loading encoders
with open('gender_encoder_le.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('geography_encoder_ohe.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('sscaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction using ANN')

# User input
geography = st.selectbox('Geography', geo_encoder.categories_[0])
age = st.slider('Age', 18, 92)
gender = st.selectbox('Gender', gender_encoder.classes_)
credit_score = st.slider('Credit Score', 300, 900)
balance = st.number_input('Balance')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])
est_salary = st.number_input('Estimate Salary')

# Prepare input dataframe
input_df = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [gender_encoder.transform([gender])[0]],  # Use encoded gender
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [est_salary]
    }
)

# One-hot encode 'Geography'
geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

# Concatenate the encoded 'Geography' columns to the input dataframe
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input_df
input_df_scaled = scaler.transform(input_df)

# Predict churn
prediction = model.predict(input_df_scaled)
prediction_probab = prediction[0][0]

# Display the result
if prediction_probab > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is likely to stay")
