import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("random_forest_model.joblib")

st.title("Tourism Rating Prediction")

st.write("Enter feature values for prediction")

# Get exact feature names from trained model
feature_names = model.feature_names_in_

# Store inputs
values = []

# Create dynamic input fields
for feature in feature_names:

    value = st.number_input(
        f"{feature}",
        value=0.0
    )

    values.append(value)

# Create dataframe EXACTLY matching training format
input_data = pd.DataFrame(
    np.array(values).reshape(1, -1),
    columns=feature_names
)

# Predict button
if st.button("Predict"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Rating: {prediction[0]}")