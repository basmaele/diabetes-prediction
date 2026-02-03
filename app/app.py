import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.title("Diabetes Predictor")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

MODEL_PATH = Path("../models/model.pkl")
model = joblib.load(MODEL_PATH)

with tab1:
    st.header("Predict Diabetes for a Single Patient")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age (years)", min_value = 0.0, max_value=150.0, step=0.1)
    hypertension = st.selectbox("Hypertension", ["No Hypertension", "Hypertension"])
    heart_diasease = st.selectbox("Heart disease", ["No Heart Disease", "Heart Disease"])
    smoking_history = st.selectbox("Smoking History", ["Never", "Ever", "Former", "Current", "Not current", "No info"])
    bmi = st.number_input("BMI (kg/m2)", min_value = 5.0, max_value=100.0, step=0.1)
    hba1c = st.number_input("HbA1c (%)", min_value = 2.0, max_value=50.0, step=0.1)
    blood_gl = st.number_input("Blood Glucose (mg/dL)", min_value = 10, max_value=1500)

    if st.button("Predict"):
        # Map input to what the model expects
        gender_map = {"Male": "Male", "Female": "Female", "Other": "Other"}
        hypertension_map = {"No Hypertension": 0, "Hypertension": 1}
        heart_disease_map = {"No Heart Disease": 0, "Heart Disease": 1}
        smoking_map = {
            "Never": "never",
            "Ever": "ever",
            "Former": "former",
            "Current": "current",
            "Not current": "not current",
            "No info": "No Info"
        }

        # Create a Dataframe with user inputs
        input_df = pd.DataFrame({
            "gender": [gender_map[gender]],
            "age": [age],
            "hypertension": [hypertension_map[hypertension]],
            "heart_disease": [heart_disease_map[heart_diasease]],
            "smoking_history": [smoking_map[smoking_history]],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [blood_gl]
        })

        # Prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0, 1]

        # Show results
        st.success(f"Predicted class (diabetes): {prediction}")
        st.info(f"Predicted probability of diabetes: {prediction_proba:.2f}")
# --- TAB 2: Bulk Predict ---
with tab2:
    st.header("Predict Diabetes from CSV file")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Input Data:", df.head())

        # Predictions
        predictions = model.predict(df)
        predictions_proba = model.predict_proba(df)[:, 1]

        df["Predicted_Diabetes"] = predictions
        df["Predicted_Probability"] = predictions_proba

        st.write("Predictions:", df.head())

        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

# --- TAB 3: Model Information ---
with tab3:
    st.header("Model Information")
    st.write("Model type:", type(model.named_steps["classifier"]))
    st.write("Pipeline steps:", model.named_steps.keys())
    st.write("Categorical features encoded:", model.named_steps["preprocessor"].transformers_[1][2])
    st.write("Numerical features scaled:", model.named_steps["preprocessor"].transformers_[0][2])
