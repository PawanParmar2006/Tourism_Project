
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('random_forest_model.joblib')

st.title("Tourism Rating Prediction System")

st.header("Enter Input Features")

# Input fields
VisitYear = st.number_input("Visit Year")

VisitMonth = st.number_input("Visit Month")

VisitMode = st.number_input("Visit Mode")

AttractionPopularity = st.number_input("Attraction Popularity")

AvgUserRating = st.number_input("Average User Rating")

UserVisitCount = st.number_input("User Visit Count")

# Create dataframe with SAME feature names
input_data = pd.DataFrame({

    'VisitYear':[VisitYear],

    'VisitMonth':[VisitMonth],

    'VisitMode':[VisitMode],

    'AttractionPopularity':[AttractionPopularity],

    'AvgUserRating':[AvgUserRating],

    'UserVisitCount':[UserVisitCount]

})

# Prediction
prediction = model.predict(input_data)

st.subheader("Predicted Rating")

st.write(prediction[0])
