import streamlit as st
import joblib
import numpy as np
import os

# Page configuration
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load the saved model and scaler
# Using os.path.join ensures it works on both Windows and Linux (Render/Streamlit Cloud)
model_path = os.path.join('model', 'house_price_model.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')

@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_assets():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'model_building.ipynb' has been run.")

# UI Elements
st.title("🏠 House Price Prediction System")
st.write("Enter the house details below to estimate the Sale Price.")

# Layout with columns for inputs
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1500)
    total_bsmt = st.number_input("Total Basement (sqft)", min_value=0, max_value=6000, value=1000)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4, 5])
    full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4])
    year_built = st.number_input("Year Built", min_value=1872, max_value=2026, value=2000)

# Prediction Logic
if st.button("Predict House Price"):
    # Arrange features in the exact order used during training
    features = np.array([[overall_qual, gr_liv_area, total_bsmt, garage_cars, full_bath, year_built]])
    
    # Scale and Predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    st.success(f"### Estimated Sale Price: ${prediction[0]:,.2f}")